use libc::{c_char, c_int, c_uchar, c_uint, c_void, size_t};

pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUdeviceptr = usize;
pub type CUresult = c_uint;

// Stream callback type
pub type CUstreamCallback = extern "C" fn(hStream: CUstream, status: CUresult, userData: *mut c_void);

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;

// Stream creation flags
pub const CU_STREAM_DEFAULT: c_uint = 0x0;
pub const CU_STREAM_NON_BLOCKING: c_uint = 0x1;

// Event creation flags
pub const CU_EVENT_DEFAULT: c_uint = 0x0;
pub const CU_EVENT_BLOCKING_SYNC: c_uint = 0x1;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x2;

#[allow(non_snake_case)]
#[link(name = "cuda")]
extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;

    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;

    pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: size_t) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyHtoD_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const c_void,
        byteCount: size_t,
    ) -> CUresult;
    pub fn cuMemcpyDtoH_v2(
        dstHost: *mut c_void,
        srcDevice: CUdeviceptr,
        byteCount: size_t,
    ) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut size_t, total: *mut size_t) -> CUresult;
    pub fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: c_uchar, N: size_t) -> CUresult;

    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;

    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuFuncSetAttribute(hfunc: CUfunction, attrib: c_uint, value: c_int) -> CUresult;

    // Stream management
    pub fn cuStreamCreate(phStream: *mut CUstream, flags: c_uint) -> CUresult;
    pub fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
    pub fn cuStreamQuery(hStream: CUstream) -> CUresult;
    pub fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, flags: c_uint) -> CUresult;
    pub fn cuStreamAddCallback(
        hStream: CUstream,
        callback: CUstreamCallback,
        userData: *mut c_void,
        flags: c_uint,
    ) -> CUresult;

    // Event management
    pub fn cuEventCreate(phEvent: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;
    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(hEvent: CUevent) -> CUresult;
    pub fn cuEventQuery(hEvent: CUevent) -> CUresult;
    pub fn cuEventElapsedTime(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent)
        -> CUresult;

    // Async memory operations
    pub fn cuMemcpyHtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const c_void,
        byteCount: size_t,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(
        dstHost: *mut c_void,
        srcDevice: CUdeviceptr,
        byteCount: size_t,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        byteCount: size_t,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD8Async(
        dstDevice: CUdeviceptr,
        uc: c_uchar,
        N: size_t,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult;
}
