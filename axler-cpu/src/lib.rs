use axler_traits::{Device, KernelHandle, KernelInfo, Renderer};
use axler_uop::{Buffer, UOp};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::ffi::c_void;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use crate::renderer::CpuRenderer;

pub mod renderer;

// Wrapper to make raw pointers Send (we ensure thread safety with mutex)
struct SendPtr(*mut c_void);
unsafe impl Send for SendPtr {}

pub struct CpuDevice {
    renderer: CpuRenderer,
    cache_dir: PathBuf,
    loaded_libraries: FxHashMap<String, libloading::Library>,
    kernel_handles: FxHashMap<String, (usize, Vec<usize>, axler_uop::DType, usize)>,
    uop_to_source_cache: FxHashMap<u64, String>,
    memory_pool: FxHashMap<(usize, axler_uop::DType), Vec<SendPtr>>,
}

impl CpuDevice {
    pub fn release_buffer(&mut self, buffer: *mut c_void, size: usize, dtype: axler_uop::DType) {
        if !buffer.is_null() {
            let key = (size, dtype);
            let pool = self.memory_pool.entry(key).or_insert_with(Vec::new);

            const MAX_POOL_SIZE: usize = 10;
            if pool.len() < MAX_POOL_SIZE {
                pool.push(SendPtr(buffer));
            }
        }
    }

    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("axler")
            .join("kernels")
            .join("cpu");

        fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        Self {
            renderer: CpuRenderer::new(),
            cache_dir,
            loaded_libraries: FxHashMap::default(),
            kernel_handles: FxHashMap::default(),
            uop_to_source_cache: FxHashMap::default(),
            memory_pool: FxHashMap::default(),
        }
    }

    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        Self {
            renderer: CpuRenderer::new(),
            cache_dir,
            loaded_libraries: FxHashMap::default(),
            kernel_handles: FxHashMap::default(),
            uop_to_source_cache: FxHashMap::default(),
            memory_pool: FxHashMap::default(),
        }
    }

    fn get_kernel_hash(source_code: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        source_code.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl Device for CpuDevice {
    fn compile(
        &mut self,
        source_code: &str,
        kernel_info: &KernelInfo,
    ) -> Result<KernelHandle, String> {
        let kernel_hash = Self::get_kernel_hash(source_code);

        let kernel_name = format!("kernel_{}", kernel_hash);
        let so_file = self.cache_dir.join(format!("lib{}.so", kernel_name));

        if so_file.exists() {
            unsafe {
                let lib = libloading::Library::new(&so_file)
                    .map_err(|e| format!("Failed to load cached library: {}", e))?;

                type KernelFn = unsafe extern "C" fn(buffers: *const *mut c_void);
                let func: libloading::Symbol<KernelFn> = lib
                    .get(b"execute_kernel")
                    .map_err(|e| format!("Failed to get function: {}", e))?;

                let func_ptr = *func as KernelHandle;

                self.loaded_libraries.insert(kernel_name.clone(), lib);
                self.kernel_handles.insert(
                    kernel_hash.clone(),
                    (
                        func_ptr as usize,
                        kernel_info.output_shape.clone(),
                        kernel_info.output_dtype,
                        kernel_info.output_size,
                    ),
                );

                return Ok(func_ptr);
            }
        }

        let c_file = self.cache_dir.join(format!("{}.c", kernel_hash));

        fs::write(&c_file, source_code)
            .map_err(|e| format!("Failed to write C source file: {}", e))?;

        let output = Command::new("gcc")
            .args(&[
                "-shared",
                "-fPIC",
                "-O3",
                "-march=native",
                "-mtune=native",
                "-ffast-math",
                "-funroll-loops",
                "-ftree-vectorize",
                "-fomit-frame-pointer",
                "-flto",
                "-fno-stack-protector",
                "-fno-math-errno",
                "-fno-trapping-math",
                "-fno-signaling-nans",
                "-o",
                so_file.to_str().unwrap(),
                c_file.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("Failed to compile: {}", e))?;

        if !output.status.success() {
            let _ = fs::remove_file(&c_file);
            let _ = fs::remove_file(&so_file);

            return Err(format!(
                "Compilation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let _ = fs::remove_file(&c_file);

        unsafe {
            let lib = libloading::Library::new(&so_file)
                .map_err(|e| format!("Failed to load shared library: {}", e))?;

            type KernelFn = unsafe extern "C" fn(buffers: *const *mut c_void);
            let func: libloading::Symbol<KernelFn> = lib
                .get(b"execute_kernel")
                .map_err(|e| format!("Failed to get function: {}", e))?;

            let func_ptr = func.into_raw().into_raw() as KernelHandle;

            self.loaded_libraries.insert(kernel_name, lib);
            self.kernel_handles.insert(
                kernel_hash,
                (
                    func_ptr as usize,
                    kernel_info.output_shape.clone(),
                    kernel_info.output_dtype,
                    kernel_info.output_size,
                ),
            );

            Ok(func_ptr)
        }
    }

    fn execute(&mut self, kernel: KernelHandle, buffers: Vec<*mut c_void>) {
        unsafe {
            type KernelFn = unsafe extern "C" fn(buffers: *const *mut c_void);
            let func = std::mem::transmute::<KernelHandle, KernelFn>(kernel);
            let buffer_ptrs = buffers.as_ptr();
            func(buffer_ptrs);
        }
    }

    fn allocate(&mut self, size: usize, dtype: axler_uop::DType) -> *mut c_void {
        let key = (size, dtype);
        if let Some(pool) = self.memory_pool.get_mut(&key) {
            if let Some(SendPtr(ptr)) = pool.pop() {
                // Don't clear the buffer - it will be overwritten by the kernel
                // This saves significant memset overhead
                return ptr;
            }
        }

        match dtype {
            axler_uop::DType::F32 => {
                let mut buf = Vec::<f32>::with_capacity(size);
                unsafe {
                    buf.set_len(size);
                }
                let ptr = buf.as_mut_ptr() as *mut c_void;
                std::mem::forget(buf);
                ptr
            }
            axler_uop::DType::U32 => {
                let mut buf = Vec::<u32>::with_capacity(size);
                unsafe {
                    buf.set_len(size);
                }
                let ptr = buf.as_mut_ptr() as *mut c_void;
                std::mem::forget(buf);
                ptr
            }
            axler_uop::DType::U8 => {
                let mut buf = Vec::<u8>::with_capacity(size);
                unsafe {
                    buf.set_len(size);
                }
                let ptr = buf.as_mut_ptr() as *mut c_void;
                std::mem::forget(buf);
                ptr
            }
        }
    }

    fn copy_host_device(
        &mut self,
        host_ptr: *const c_void,
        device_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    ) {
        unsafe {
            let bytes = match dtype {
                axler_uop::DType::F32 => size * std::mem::size_of::<f32>(),
                axler_uop::DType::U32 => size * std::mem::size_of::<u32>(),
                axler_uop::DType::U8 => size * std::mem::size_of::<u8>(),
            };
            std::ptr::copy_nonoverlapping(host_ptr as *const u8, device_ptr as *mut u8, bytes);
        }
    }

    fn copy_device_host(
        &self,
        device_ptr: *const c_void,
        host_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    ) {
        unsafe {
            let bytes = match dtype {
                axler_uop::DType::F32 => size * std::mem::size_of::<f32>(),
                axler_uop::DType::U32 => size * std::mem::size_of::<u32>(),
                axler_uop::DType::U8 => size * std::mem::size_of::<u8>(),
            };
            std::ptr::copy_nonoverlapping(device_ptr as *const u8, host_ptr as *mut u8, bytes);
        }
    }

    fn deallocate(&mut self, ptr: *mut c_void, size: usize, dtype: axler_uop::DType) {
        if ptr.is_null() {
            return;
        }

        self.release_buffer(ptr, size, dtype);
    }

    fn get_or_compile_kernel(
        &mut self,
        uop: &UOp,
        buffers: &[Buffer],
    ) -> (KernelHandle, Vec<usize>, axler_uop::DType, usize) {
        let uop_hash = uop.compute_hash();

        if let Some(source_hash) = self.uop_to_source_cache.get(&uop_hash) {
            if let Some(&(handle, ref shape, dtype, size)) = self.kernel_handles.get(source_hash) {
                return (handle as KernelHandle, shape.clone(), dtype, size);
            }
        }

        let lowered = self.renderer.lower_if_required(uop, buffers);
        let source_code = self.renderer.render(&lowered, uop);
        let source_hash = Self::get_kernel_hash(&source_code);

        self.uop_to_source_cache
            .insert(uop_hash, source_hash.clone());

        if let Some(&(handle, ref shape, dtype, size)) = self.kernel_handles.get(&source_hash) {
            return (handle as KernelHandle, shape.clone(), dtype, size);
        }

        let kernel_info = KernelInfo {
            source_code,
            output_shape: lowered.output_shape.clone(),
            output_size: lowered.output_size,
            output_dtype: lowered.output_dtype,
        };
        let handle = self
            .compile(&kernel_info.source_code, &kernel_info)
            .unwrap();
        (
            handle,
            lowered.output_shape,
            lowered.output_dtype,
            lowered.output_size,
        )
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

pub static CPU_DEVICE: Lazy<Mutex<CpuDevice>> = Lazy::new(|| Mutex::new(CpuDevice::new()));

pub fn get_cpu_device() -> &'static Mutex<CpuDevice> {
    &CPU_DEVICE
}
