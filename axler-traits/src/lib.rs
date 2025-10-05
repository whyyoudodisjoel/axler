use std::ffi::c_void;
use axler_uop::{Buffer, UOp};

pub trait Renderer<LoweredUOp> {
    fn render<'a>(&mut self, lowered_uop: &LoweredUOp, uop: &'a UOp<'a>) -> String;

    fn lower_if_required<'a>(&mut self, uop: &'a UOp<'a>, buffers: &[Buffer]) -> LoweredUOp;
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

pub trait Device {
    fn compile(
        &mut self,
        source_code: &str,
        kernel_info: &KernelInfo,
    ) -> Result<KernelHandle, String>;

    fn execute(&mut self, kernel: KernelHandle, buffers: Vec<*mut c_void>);

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
