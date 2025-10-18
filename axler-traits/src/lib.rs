use axler_uop::{Buffer, UOp};
use std::ffi::c_void;
use std::future::Future;

pub trait Renderer<LoweredUOp> {
    fn render(&mut self, lowered_uop: &LoweredUOp, uop: &UOp) -> String;

    fn lower_if_required(&mut self, uop: &UOp, buffers: &[Buffer]) -> LoweredUOp;
}

// Kernel handle - opaque type that each device can interpret differently
// For CPU: can be a function pointer
// For CUDA: can be a struct containing module and function info
pub type KernelHandle = *mut c_void;

pub struct KernelInfo {
    pub source_code: String,
    pub output_shape: Vec<usize>,
    pub output_size: usize,
    pub output_dtype: axler_uop::DType,
}

/// Synchronous device operations
pub trait Device {
    fn compile(
        &mut self,
        source_code: &str,
        kernel_info: &KernelInfo,
    ) -> Result<KernelHandle, String>;

    fn execute(&mut self, kernel: KernelHandle, buffers: Vec<*mut c_void>);

    /// Get a cached kernel or compile it if not in cache
    /// Returns (kernel_handle, output_shape, output_dtype, output_size)
    fn get_or_compile_kernel(
        &mut self,
        uop: &UOp,
        buffers: &[Buffer],
    ) -> (KernelHandle, Vec<usize>, axler_uop::DType, usize);

    // Memory management
    fn allocate(&mut self, size: usize, dtype: axler_uop::DType) -> *mut c_void;

    fn copy_host_device(
        &mut self,
        host_ptr: *const c_void,
        device_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    );

    fn copy_device_host(
        &self,
        device_ptr: *const c_void,
        host_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    );

    fn deallocate(&mut self, ptr: *mut c_void, size: usize, dtype: axler_uop::DType);

    fn free_kernel(&mut self, kernel: KernelHandle) {
        let _ = kernel;
    }
}

/// Asynchronous device operations
/// Devices that support async execution should implement this trait
pub trait AsyncDevice: Device {
    type DeviceFuture: Future<Output = Result<(), String>>;

    fn execute_async(
        &mut self,
        kernel: KernelHandle,
        buffers: Vec<*mut c_void>,
    ) -> Result<Self::DeviceFuture, String>;

    /// Async memory allocation (returns immediately, actual allocation may be deferred)
    fn allocate_async(&mut self, size: usize, dtype: axler_uop::DType) -> *mut c_void {
        // Default to synchronous allocation
        self.allocate(size, dtype)
    }

    /// Async host-to-device copy, returns a future
    fn copy_host_device_async(
        &mut self,
        host_ptr: *const c_void,
        device_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    ) -> Result<Self::DeviceFuture, String> {
        // Default to synchronous copy and return completed future
        self.copy_host_device(host_ptr, device_ptr, size, dtype);
        Err("Async copy not supported for this device".to_string())
    }

    /// Async device-to-host copy, returns a future
    fn copy_device_host_async(
        &self,
        device_ptr: *const c_void,
        host_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    ) -> Result<Self::DeviceFuture, String> {
        // Default to synchronous copy
        self.copy_device_host(device_ptr, host_ptr, size, dtype);
        Err("Async copy not supported for this device".to_string())
    }

    /// Async deallocation (may be deferred)
    fn deallocate_async(&mut self, ptr: *mut c_void, size: usize, dtype: axler_uop::DType) {
        // Default to synchronous deallocation
        self.deallocate(ptr, size, dtype);
    }
}
