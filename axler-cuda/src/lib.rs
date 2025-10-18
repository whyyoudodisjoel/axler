pub mod async_ops;
pub mod ffi;
pub mod renderer;

use axler_traits::{Device, KernelHandle, KernelInfo, Renderer};
use once_cell::sync::Lazy;
use rustc_hash::{FxHashMap, FxHasher};
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use crate::async_ops::{AsyncCudaContext, CudaFuture};
use crate::ffi::nvrtc::compile_cuda_to_ptx;
use crate::ffi::{CudaContext, CudaFunction, CudaModule, CudaStream};
use crate::renderer::CudaRenderer;
use std::sync::Arc;

// Wrapper to make raw device pointers Send (we ensure thread safety with mutex)
// Stores raw CUdeviceptr for manual memory management
struct SendDevicePtr(usize);
unsafe impl Send for SendDevicePtr {}

pub struct CudaDevice {
    context: CudaContext,
    renderer: CudaRenderer,
    max_threads_per_block: u32,
    max_grid_size: (u32, u32, u32),
    // Store loaded kernels by function pointer for O(1) execution lookup
    loaded_kernels: FxHashMap<usize, (CudaModule, CudaFunction, usize)>,
    // Cache source hash -> (function ptr, shape, dtype, size) - same as CPU
    kernel_cache: FxHashMap<String, (usize, Vec<usize>, axler_uop::DType, usize)>,
    // Cache UOp hash -> source code hash mapping to skip rendering
    uop_to_source_cache: FxHashMap<u64, String>,
    // Memory pool for reusing allocations (size -> list of free buffers)
    memory_pool: FxHashMap<(usize, axler_uop::DType), Vec<SendDevicePtr>>,
    // Async context for stream-based operations
    async_context: Option<Arc<AsyncCudaContext>>,
}

impl CudaDevice {
    fn release_buffer(&mut self, ptr: usize, size: usize, dtype: axler_uop::DType) {
        let key = (size, dtype);
        let pool = self.memory_pool.entry(key).or_insert_with(Vec::new);

        // Only keep a reasonable number of buffers in the pool to avoid excessive memory use
        const MAX_POOL_SIZE: usize = 10;
        if pool.len() < MAX_POOL_SIZE {
            pool.push(SendDevicePtr(ptr));
        } else {
            // Free the memory if pool is full
            use crate::ffi::cuMemFree_v2;
            unsafe {
                let _ = cuMemFree_v2(ptr);
            }
        }
    }

    /// Clear all cached buffers from the memory pool to free GPU memory
    pub fn clear_memory_pool(&mut self) {
        use crate::ffi::cuMemFree_v2;

        // Free all pooled buffers
        for (_, pool) in self.memory_pool.iter() {
            for SendDevicePtr(ptr) in pool {
                unsafe {
                    let _ = cuMemFree_v2(*ptr);
                }
            }
        }

        self.memory_pool.clear();
    }

    fn calculate_bytes(size: usize, dtype: axler_uop::DType) -> usize {
        match dtype {
            axler_uop::DType::F32 => size * std::mem::size_of::<f32>(),
            axler_uop::DType::U32 => size * std::mem::size_of::<u32>(),
            axler_uop::DType::U8 => size * std::mem::size_of::<u8>(),
        }
    }

    fn ensure_context_current(&self) -> Result<(), String> {
        self.context.set_current()
    }

    pub fn new() -> Result<Self, String> {
        let context = CudaContext::new(0)?;

        // Default to common values for modern GPUs
        let max_threads_per_block = 1024;
        let max_grid_size = (2147483647, 65535, 65535); // Common limits

        Ok(Self {
            context,
            renderer: CudaRenderer::new(),
            max_threads_per_block,
            max_grid_size,
            loaded_kernels: FxHashMap::default(),
            kernel_cache: FxHashMap::default(),
            uop_to_source_cache: FxHashMap::default(),
            memory_pool: FxHashMap::default(),
            async_context: None,
        })
    }

    /// Enable async operations with the specified number of streams
    pub fn enable_async(&mut self, num_streams: usize) -> Result<(), String> {
        self.async_context = Some(Arc::new(AsyncCudaContext::new(num_streams)?));
        Ok(())
    }

    /// Get the async context (if enabled)
    pub fn async_context(&self) -> Option<&Arc<AsyncCudaContext>> {
        self.async_context.as_ref()
    }

    /// Execute a kernel asynchronously on a specific stream
    fn execute_async_on_stream(
        &mut self,
        kernel: KernelHandle,
        mut buffers: Vec<*mut c_void>,
        stream: Arc<CudaStream>,
    ) -> Result<CudaFuture, String> {
        self.ensure_context_current()
            .expect("Failed to set CUDA context as current");

        let func_ptr = kernel as usize;
        let (_, function, output_size) = self
            .loaded_kernels
            .get(&func_ptr)
            .expect("Invalid kernel handle");

        let (grid_size, block_size) = self.calculate_optimal_launch_config(*output_size);

        function
            .launch_async(
                grid_size,
                block_size,
                0, // shared memory
                &mut buffers,
                &stream,
            )
            .expect("Failed to launch kernel asynchronously");

        Ok(CudaFuture::new(stream))
    }

    /// Execute a kernel asynchronously with auto-selected stream from context
    pub async fn execute_async_auto(
        &mut self,
        kernel: KernelHandle,
        buffers: Vec<*mut c_void>,
    ) -> Result<CudaFuture, String> {
        let stream = self
            .async_context
            .as_ref()
            .ok_or("Async context not enabled. Call enable_async() first")?
            .get_stream();

        self.execute_async_on_stream(kernel, buffers, stream)
    }

    /// Async copy from host to device

    /// Synchronize all async streams
    pub async fn synchronize_async(&self) -> Result<(), String> {
        if let Some(ctx) = &self.async_context {
            ctx.synchronize_all().await
        } else {
            Ok(())
        }
    }

    fn get_kernel_hash(source_code: &str) -> String {
        let mut hasher = FxHasher::default();
        source_code.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    // TODO: Needs a better rewrite
    fn calculate_optimal_launch_config(
        &self,
        output_size: usize,
    ) -> ((u32, u32, u32), (u32, u32, u32)) {
        let total_elements = output_size as u32;

        let threads_per_block = 256u32.min(self.max_threads_per_block);

        let blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;

        let grid_x = blocks_needed.min(self.max_grid_size.0);

        let grid_y = if blocks_needed > self.max_grid_size.0 {
            ((blocks_needed + self.max_grid_size.0 - 1) / self.max_grid_size.0)
                .min(self.max_grid_size.1)
        } else {
            1
        };

        let grid_z = if blocks_needed > grid_x * grid_y {
            ((blocks_needed + (grid_x * grid_y) - 1) / (grid_x * grid_y)).min(self.max_grid_size.2)
        } else {
            1
        };

        let grid_size = (grid_x, grid_y, grid_z);
        let block_size = (threads_per_block, 1, 1);

        (grid_size, block_size)
    }
}

impl Device for CudaDevice {
    fn compile(
        &mut self,
        source_code: &str,
        kernel_info: &KernelInfo,
    ) -> Result<KernelHandle, String> {
        let kernel_hash = Self::get_kernel_hash(source_code);

        // IMPORTANT: compile() is ONLY called from get_or_compile_kernel() when
        // the kernel is NOT in kernel_cache. So we skip that check entirely.

        self.ensure_context_current()
            .map_err(|e| format!("Failed to set CUDA context: {}", e))?;

        let kernel_name = format!("kernel_{}", kernel_hash);

        // Compile CUDA to PTX
        let ptx = compile_cuda_to_ptx(source_code, &kernel_name)?;

        let module = CudaModule::from_ptx(&ptx)?;

        let function = module.get_function("kernel")?;

        // Get function pointer to use as handle (same as CPU)
        let func_ptr = function.as_ptr() as usize;

        // Store module, function and output size for fast execution
        self.loaded_kernels
            .insert(func_ptr, (module, function, kernel_info.output_size));

        // Store handle and metadata (same structure as CPU)
        self.kernel_cache.insert(
            kernel_hash,
            (
                func_ptr,
                kernel_info.output_shape.clone(),
                kernel_info.output_dtype,
                kernel_info.output_size,
            ),
        );

        Ok(func_ptr as KernelHandle)
    }

    fn execute(&mut self, kernel: KernelHandle, mut buffers: Vec<*mut c_void>) {
        self.ensure_context_current()
            .expect("Failed to set CUDA context as current");

        // Direct lookup - O(1) instead of searching through all values
        let func_ptr = kernel as usize;
        let (_, function, output_size) = self
            .loaded_kernels
            .get(&func_ptr)
            .expect("Invalid kernel handle");

        let (grid_size, block_size) = self.calculate_optimal_launch_config(*output_size);

        function
            .launch(
                grid_size,
                block_size,
                0, // shared memory
                &mut buffers,
            )
            .expect("Failed to launch kernel");

        self.context.synchronize().expect("Failed to synchronize");
    }

    fn allocate(&mut self, size: usize, dtype: axler_uop::DType) -> *mut c_void {
        self.ensure_context_current()
            .expect("Failed to set CUDA context as current");

        // Try to get a buffer from the pool first
        let key = (size, dtype);
        if let Some(pool) = self.memory_pool.get_mut(&key) {
            if let Some(SendDevicePtr(ptr)) = pool.pop() {
                // Don't clear the buffer - it will be overwritten by the kernel
                // This saves significant memset overhead on GPU
                return ptr as *mut c_void;
            }
        }

        // No buffer available in pool, allocate new one
        let bytes = Self::calculate_bytes(size, dtype);
        use crate::ffi::cuMemAlloc_v2;

        let mut ptr: usize = 0;
        unsafe {
            crate::ffi::check_cuda_error(cuMemAlloc_v2(&mut ptr, bytes))
                .expect("Failed to allocate device memory");
        }

        ptr as *mut c_void
    }

    fn copy_host_device(
        &mut self,
        host_ptr: *const c_void,
        device_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    ) {
        self.ensure_context_current()
            .expect("Failed to set CUDA context as current");
        let bytes = Self::calculate_bytes(size, dtype);

        use crate::ffi::cuMemcpyHtoD_v2;
        unsafe {
            crate::ffi::check_cuda_error(cuMemcpyHtoD_v2(device_ptr as usize, host_ptr, bytes))
                .expect("Failed to copy from host to device");
        }
    }

    fn copy_device_host(
        &self,
        device_ptr: *const c_void,
        host_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    ) {
        self.ensure_context_current()
            .expect("Failed to set CUDA context as current");
        let bytes = Self::calculate_bytes(size, dtype);

        use crate::ffi::cuMemcpyDtoH_v2;
        unsafe {
            crate::ffi::check_cuda_error(cuMemcpyDtoH_v2(host_ptr, device_ptr as usize, bytes))
                .expect("Failed to copy from device to host");
        }
    }

    fn deallocate(&mut self, ptr: *mut c_void, size: usize, dtype: axler_uop::DType) {
        if ptr.is_null() {
            return;
        }

        self.ensure_context_current()
            .expect("Failed to set CUDA context as current");

        // Return buffer to pool
        self.release_buffer(ptr as usize, size, dtype);
    }

    fn free_kernel(&mut self, kernel: KernelHandle) {
        let func_ptr = kernel as usize;

        // Remove from loaded kernels
        self.loaded_kernels.remove(&func_ptr);

        // Remove from kernel cache
        self.kernel_cache
            .retain(|_, &mut (ptr, _, _, _)| ptr != func_ptr);
    }

    fn get_or_compile_kernel(
        &mut self,
        uop: &axler_uop::UOp,
        buffers: &[axler_uop::Buffer],
    ) -> (KernelHandle, Vec<usize>, axler_uop::DType, usize) {
        use axler_traits::Renderer;

        // Compute UOp hash
        let uop_hash = uop.compute_hash();

        // Check if we have a cached source code hash for this UOp
        if let Some(source_hash) = self.uop_to_source_cache.get(&uop_hash) {
            // We've seen this UOp before, check if kernel is compiled
            if let Some(&(handle, ref shape, dtype, size)) = self.kernel_cache.get(source_hash) {
                return (handle as KernelHandle, shape.clone(), dtype, size);
            }
        }

        // Need to render and compile
        let lowered = self.renderer.lower_if_required(uop, buffers);
        let source_code = self.renderer.render(&lowered, uop);
        let source_hash = Self::get_kernel_hash(&source_code);

        // Cache the UOp -> source mapping
        self.uop_to_source_cache
            .insert(uop_hash, source_hash.clone());

        // Check if this source code was already compiled (different UOp, same source)
        if let Some(&(handle, ref shape, dtype, size)) = self.kernel_cache.get(&source_hash) {
            return (handle as KernelHandle, shape.clone(), dtype, size);
        }

        // Not cached, need to compile
        let kernel_info = axler_traits::KernelInfo {
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

impl axler_traits::AsyncDevice for CudaDevice {
    type DeviceFuture = CudaFuture;

    fn execute_async(
        &mut self,
        kernel: KernelHandle,
        buffers: Vec<*mut c_void>,
    ) -> Result<Self::DeviceFuture, String> {
        // Get stream from async context, or create a new one
        let stream = if let Some(ctx) = &self.async_context {
            ctx.get_stream()
        } else {
            // If no async context, create a single stream
            Arc::new(CudaStream::new()?)
        };

        // Reuse the existing helper method
        self.execute_async_on_stream(kernel, buffers, stream)
    }

    fn allocate_async(&mut self, size: usize, dtype: axler_uop::DType) -> *mut c_void {
        // For now, CUDA allocation is fast enough to be synchronous
        // In the future, could use memory pools or stream-ordered allocations
        self.allocate(size, dtype)
    }

    fn copy_host_device_async(
        &mut self,
        host_ptr: *const c_void,
        device_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    ) -> Result<Self::DeviceFuture, String> {
        self.ensure_context_current()
            .map_err(|e| format!("Failed to set CUDA context: {}", e))?;

        let bytes = Self::calculate_bytes(size, dtype);

        // Get stream from async context
        let stream = if let Some(ctx) = &self.async_context {
            ctx.get_stream()
        } else {
            Arc::new(CudaStream::new()?)
        };

        use crate::ffi::cuMemcpyHtoDAsync_v2;
        unsafe {
            crate::ffi::check_cuda_error(cuMemcpyHtoDAsync_v2(
                device_ptr as usize,
                host_ptr,
                bytes,
                stream.raw(),
            ))
            .map_err(|e| format!("Failed to async copy from host to device: {}", e))?;
        }

        Ok(CudaFuture::new(stream))
    }

    fn copy_device_host_async(
        &self,
        device_ptr: *const c_void,
        host_ptr: *mut c_void,
        size: usize,
        dtype: axler_uop::DType,
    ) -> Result<Self::DeviceFuture, String> {
        self.ensure_context_current()
            .map_err(|e| format!("Failed to set CUDA context: {}", e))?;

        let bytes = Self::calculate_bytes(size, dtype);

        // Get stream from async context
        let stream = if let Some(ctx) = &self.async_context {
            ctx.get_stream()
        } else {
            Arc::new(CudaStream::new()?)
        };

        use crate::ffi::cuMemcpyDtoHAsync_v2;
        unsafe {
            crate::ffi::check_cuda_error(cuMemcpyDtoHAsync_v2(
                host_ptr,
                device_ptr as usize,
                bytes,
                stream.raw(),
            ))
            .map_err(|e| format!("Failed to async copy from device to host: {}", e))?;
        }

        Ok(CudaFuture::new(stream))
    }

    fn deallocate_async(&mut self, ptr: *mut c_void, size: usize, dtype: axler_uop::DType) {
        // Deallocation can be done synchronously as it's just returning to the pool
        self.deallocate(ptr, size, dtype);
    }
}

pub static CUDA_DEVICE: Lazy<Mutex<Option<CudaDevice>>> = Lazy::new(|| match CudaDevice::new() {
    Ok(device) => Mutex::new(Some(device)),
    Err(_) => Mutex::new(None),
});

pub fn get_cuda_device() -> Option<&'static Mutex<Option<CudaDevice>>> {
    if CUDA_DEVICE.lock().unwrap().is_some() {
        Some(&CUDA_DEVICE)
    } else {
        None
    }
}
