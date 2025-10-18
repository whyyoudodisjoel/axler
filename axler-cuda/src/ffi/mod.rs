mod error;
pub mod nvrtc;
pub mod raw;

pub use error::*;
pub use nvrtc::*;
pub use raw::*;

use std::ffi::c_void;
use std::ptr;

pub struct CudaContext {
    context: CUcontext,
    device: CUdevice,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self, String> {
        init_cuda()?;
        let device = get_device(device_id)?;
        let context = create_context(device)?;
        Ok(Self { context, device })
    }

    pub fn set_current(&self) -> Result<(), String> {
        unsafe { check_cuda_error(cuCtxSetCurrent(self.context)) }
    }

    pub fn synchronize(&self) -> Result<(), String> {
        unsafe { check_cuda_error(cuCtxSynchronize()) }
    }

    pub fn device(&self) -> CUdevice {
        self.device
    }

    pub fn raw(&self) -> CUcontext {
        self.context
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            let _ = cuCtxDestroy_v2(self.context);
        }
    }
}

pub struct CudaModule {
    module: CUmodule,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    pub fn from_ptx(ptx: &[u8]) -> Result<Self, String> {
        let mut module: CUmodule = ptr::null_mut();
        unsafe {
            check_cuda_error(cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void))?;
        }
        Ok(Self { module })
    }

    pub fn get_function(&self, name: &str) -> Result<CudaFunction, String> {
        let mut func: CUfunction = ptr::null_mut();
        let name_cstring = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
        unsafe {
            check_cuda_error(cuModuleGetFunction(
                &mut func,
                self.module,
                name_cstring.as_ptr(),
            ))?;
        }
        Ok(CudaFunction { func })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            let _ = cuModuleUnload(self.module);
        }
    }
}

pub struct CudaFunction {
    func: CUfunction,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaFunction {
    pub fn as_ptr(&self) -> *const c_void {
        self.func as *const c_void
    }

    pub fn launch(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        unsafe {
            // For cuLaunchKernel, we need an array of pointers to the parameters
            // Each element in params is a device pointer that should be passed to the kernel
            // We need to create an array of pointers to these pointers
            let mut param_ptrs: Vec<*mut c_void> = params
                .iter_mut()
                .map(|p| p as *mut *mut c_void as *mut c_void)
                .collect();

            check_cuda_error(cuLaunchKernel(
                self.func,
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                shared_mem_bytes,
                ptr::null_mut(), // stream
                param_ptrs.as_mut_ptr(),
                ptr::null_mut(), // extra
            ))
        }
    }

    pub fn launch_async(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
        stream: &CudaStream,
    ) -> Result<(), String> {
        unsafe {
            let mut param_ptrs: Vec<*mut c_void> = params
                .iter_mut()
                .map(|p| p as *mut *mut c_void as *mut c_void)
                .collect();

            check_cuda_error(cuLaunchKernel(
                self.func,
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                shared_mem_bytes,
                stream.raw(),
                param_ptrs.as_mut_ptr(),
                ptr::null_mut(), // extra
            ))
        }
    }

    pub fn set_attribute(&self, attrib: u32, value: i32) -> Result<(), String> {
        unsafe { check_cuda_error(cuFuncSetAttribute(self.func, attrib, value)) }
    }
}

pub fn init_cuda() -> Result<(), String> {
    unsafe { check_cuda_error(cuInit(0)) }
}

pub fn get_device_count() -> Result<i32, String> {
    let mut count: i32 = 0;
    unsafe {
        check_cuda_error(cuDeviceGetCount(&mut count))?;
    }
    Ok(count)
}

pub fn get_device(ordinal: i32) -> Result<CUdevice, String> {
    let mut device: CUdevice = 0;
    unsafe {
        check_cuda_error(cuDeviceGet(&mut device, ordinal))?;
    }
    Ok(device)
}

pub fn create_context(device: CUdevice) -> Result<CUcontext, String> {
    let mut context: CUcontext = ptr::null_mut();
    unsafe {
        check_cuda_error(cuCtxCreate_v2(&mut context, 0, device))?;
    }
    Ok(context)
}

pub fn get_memory_info() -> Result<(usize, usize), String> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        check_cuda_error(cuMemGetInfo_v2(&mut free, &mut total))?;
    }
    Ok((free, total))
}

pub struct CudaStream {
    stream: CUstream,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self, String> {
        let mut stream: CUstream = ptr::null_mut();
        unsafe {
            check_cuda_error(cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING))?;
        }
        Ok(Self { stream })
    }

    pub fn new_default() -> Result<Self, String> {
        let mut stream: CUstream = ptr::null_mut();
        unsafe {
            check_cuda_error(cuStreamCreate(&mut stream, CU_STREAM_DEFAULT))?;
        }
        Ok(Self { stream })
    }

    pub fn synchronize(&self) -> Result<(), String> {
        unsafe { check_cuda_error(cuStreamSynchronize(self.stream)) }
    }

    pub fn query(&self) -> Result<bool, String> {
        unsafe {
            match cuStreamQuery(self.stream) {
                CUDA_SUCCESS => Ok(true),
                CUDA_ERROR_NOT_READY => Ok(false),
                err => {
                    check_cuda_error(err)?;
                    Ok(false)
                }
            }
        }
    }

    pub fn is_ready(&self) -> bool {
        self.query().unwrap_or(false)
    }

    pub fn raw(&self) -> CUstream {
        self.stream
    }

    pub fn wait_event(&self, event: &CudaEvent) -> Result<(), String> {
        unsafe { check_cuda_error(cuStreamWaitEvent(self.stream, event.raw(), 0)) }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                let _ = cuStreamDestroy_v2(self.stream);
            }
        }
    }
}

pub struct CudaEvent {
    event: CUevent,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    pub fn new() -> Result<Self, String> {
        let mut event: CUevent = ptr::null_mut();
        unsafe {
            check_cuda_error(cuEventCreate(&mut event, CU_EVENT_DEFAULT))?;
        }
        Ok(Self { event })
    }

    pub fn new_with_flags(flags: u32) -> Result<Self, String> {
        let mut event: CUevent = ptr::null_mut();
        unsafe {
            check_cuda_error(cuEventCreate(&mut event, flags))?;
        }
        Ok(Self { event })
    }

    pub fn record(&self, stream: &CudaStream) -> Result<(), String> {
        unsafe { check_cuda_error(cuEventRecord(self.event, stream.raw())) }
    }

    pub fn synchronize(&self) -> Result<(), String> {
        unsafe { check_cuda_error(cuEventSynchronize(self.event)) }
    }

    pub fn query(&self) -> Result<bool, String> {
        unsafe {
            match cuEventQuery(self.event) {
                CUDA_SUCCESS => Ok(true),
                CUDA_ERROR_NOT_READY => Ok(false),
                err => {
                    check_cuda_error(err)?;
                    Ok(false)
                }
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.query().unwrap_or(false)
    }

    pub fn elapsed_time(&self, start: &CudaEvent) -> Result<f32, String> {
        let mut ms: f32 = 0.0;
        unsafe {
            check_cuda_error(cuEventElapsedTime(&mut ms, start.event, self.event))?;
        }
        Ok(ms)
    }

    pub fn raw(&self) -> CUevent {
        self.event
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                let _ = cuEventDestroy_v2(self.event);
            }
        }
    }
}
