use libc::{c_char, c_int, c_void, size_t};
use std::ffi::CString;
use std::ptr;

#[allow(non_camel_case_types)]
pub type nvrtcProgram = *mut c_void;
#[allow(non_camel_case_types)]
pub type nvrtcResult = u32;

pub const NVRTC_SUCCESS: nvrtcResult = 0;
pub const NVRTC_ERROR_OUT_OF_MEMORY: nvrtcResult = 1;
pub const NVRTC_ERROR_PROGRAM_CREATION_FAILURE: nvrtcResult = 2;
pub const NVRTC_ERROR_INVALID_INPUT: nvrtcResult = 3;
pub const NVRTC_ERROR_INVALID_PROGRAM: nvrtcResult = 4;
pub const NVRTC_ERROR_INVALID_OPTION: nvrtcResult = 5;
pub const NVRTC_ERROR_COMPILATION: nvrtcResult = 6;

#[allow(non_snake_case)]
#[link(name = "nvrtc")]
extern "C" {
    pub fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        numHeaders: c_int,
        headers: *const *const c_char,
        includeNames: *const *const c_char,
    ) -> nvrtcResult;

    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;

    pub fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        numOptions: c_int,
        options: *const *const c_char,
    ) -> nvrtcResult;

    pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut size_t) -> nvrtcResult;
    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;

    pub fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut size_t) -> nvrtcResult;
    pub fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
    pub fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;
    pub fn nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> nvrtcResult;
}

pub fn check_nvrtc_error(result: nvrtcResult) -> Result<(), String> {
    if result != NVRTC_SUCCESS {
        let error_str = unsafe {
            let ptr = nvrtcGetErrorString(result);
            if !ptr.is_null() {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            } else {
                format!("Unknown NVRTC error: {}", result)
            }
        };
        Err(error_str)
    } else {
        Ok(())
    }
}

pub fn get_program_log(program: nvrtcProgram) -> Result<String, String> {
    unsafe {
        let mut log_size: size_t = 0;
        check_nvrtc_error(nvrtcGetProgramLogSize(program, &mut log_size))?;

        if log_size <= 1 {
            return Ok(String::new());
        }

        let mut log_buffer = vec![0u8; log_size];
        check_nvrtc_error(nvrtcGetProgramLog(
            program,
            log_buffer.as_mut_ptr() as *mut c_char,
        ))?;

        // Convert to string, removing null terminator
        log_buffer.pop(); // Remove null terminator
        Ok(String::from_utf8_lossy(&log_buffer).into_owned())
    }
}

pub fn compile_cuda_to_ptx(source: &str, kernel_name: &str) -> Result<Vec<u8>, String> {
    compile_cuda_to_ptx_with_options(
        source,
        kernel_name,
        &[
            "--gpu-architecture=compute_89", // RTX 4090 compute capability
            "--use_fast_math",
            "-default-device",
            "--extra-device-vectorization",
            "--ftz=true",
            "--prec-div=false",
            "--prec-sqrt=false",
            "--fmad=true",
            "--restrict",
        ],
    )
}

fn compile_cuda_to_ptx_with_options(
    source: &str,
    kernel_name: &str,
    compile_options: &[&str],
) -> Result<Vec<u8>, String> {
    unsafe {
        let mut prog: nvrtcProgram = ptr::null_mut();

        let src_cstring = CString::new(source).map_err(|e| e.to_string())?;
        let name_cstring = CString::new(kernel_name).map_err(|e| e.to_string())?;

        check_nvrtc_error(nvrtcCreateProgram(
            &mut prog,
            src_cstring.as_ptr(),
            name_cstring.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        ))?;

        // Prepare compile ptions
        let options: Vec<CString> = compile_options
            .iter()
            .map(|&opt| CString::new(opt).unwrap())
            .collect();
        let option_ptrs: Vec<*const c_char> = options.iter().map(|s| s.as_ptr()).collect();

        let compile_result =
            nvrtcCompileProgram(prog, option_ptrs.len() as c_int, option_ptrs.as_ptr());

        if compile_result != NVRTC_SUCCESS {
            let error_msg = if let Err(e) = check_nvrtc_error(compile_result) {
                format!("NVRTC compilation failed: {}", e)
            } else {
                "NVRTC compilation failed".to_string()
            };

            if let Ok(log) = get_program_log(prog) {
                let full_error = format!("{}\nCompilation log:\n{}", error_msg, log);
                nvrtcDestroyProgram(&mut prog);
                return Err(full_error);
            }

            nvrtcDestroyProgram(&mut prog);
            return Err(error_msg);
        }

        let mut ptx_size: size_t = 0;
        check_nvrtc_error(nvrtcGetPTXSize(prog, &mut ptx_size))?;

        let mut ptx = vec![0u8; ptx_size];
        check_nvrtc_error(nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut c_char))?;

        nvrtcDestroyProgram(&mut prog);

        Ok(ptx)
    }
}
