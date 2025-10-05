use super::raw::*;

pub fn check_cuda_error(result: CUresult) -> Result<(), String> {
    if result != CUDA_SUCCESS {
        let error_str = get_cuda_error_string(result);
        Err(format!("CUDA error {}: {}", result, error_str))
    } else {
        Ok(())
    }
}

fn get_cuda_error_string(error: CUresult) -> String {
    unsafe {
        let mut ptr: *const i8 = std::ptr::null();
        if cuGetErrorString(error, &mut ptr) == CUDA_SUCCESS && !ptr.is_null() {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        } else {
            match error {
                CUDA_ERROR_INVALID_VALUE => "Invalid value",
                CUDA_ERROR_OUT_OF_MEMORY => "Out of memory",
                CUDA_ERROR_NOT_INITIALIZED => "Driver not initialized",
                CUDA_ERROR_DEINITIALIZED => "Driver deinitialized",
                CUDA_ERROR_NO_DEVICE => "No CUDA-capable device",
                CUDA_ERROR_INVALID_DEVICE => "Invalid device",
                CUDA_ERROR_NOT_READY => "Not ready",
                _ => "Unknown error",
            }
            .to_string()
        }
    }
}
