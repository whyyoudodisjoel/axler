use crate::ffi::CudaStream;
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Callback data structure passed to CUDA stream callback
struct CallbackData {
    waker: Mutex<Option<Waker>>,
    completed: AtomicBool,
}

/// Represents an async CUDA operation that can be awaited
pub struct CudaFuture {
    stream: Arc<CudaStream>,
    callback_data: Arc<CallbackData>,
    callback_registered: AtomicBool,
}

impl CudaFuture {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            callback_data: Arc::new(CallbackData {
                waker: Mutex::new(None),
                completed: AtomicBool::new(false),
            }),
            callback_registered: AtomicBool::new(false),
        }
    }

    /// Register the CUDA stream callback
    fn register_callback(&self) -> Result<(), String> {
        use crate::ffi::{check_cuda_error, cuStreamAddCallback, CUresult};

        let callback_data_ptr = Arc::into_raw(self.callback_data.clone()) as *mut c_void;

        extern "C" fn cuda_callback(_stream: crate::ffi::raw::CUstream, _status: CUresult, user_data: *mut c_void) {
            unsafe {
                let callback_data = Arc::from_raw(user_data as *const CallbackData);

                // Mark as completed (atomic, no lock needed)
                callback_data.completed.store(true, Ordering::Release);

                // Wake the waker if it exists
                if let Some(waker) = callback_data.waker.lock().unwrap().take() {
                    waker.wake();
                }

                // Don't drop the Arc - we still need it in the Future
                std::mem::forget(callback_data);
            }
        }

        unsafe {
            check_cuda_error(cuStreamAddCallback(
                self.stream.raw(),
                cuda_callback,
                callback_data_ptr,
                0,
            ))
        }
    }
}

impl Future for CudaFuture {
    type Output = Result<(), String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.callback_data.completed.load(Ordering::Acquire) {
            return Poll::Ready(Ok(()));
        }

        if !self.callback_registered.swap(true, Ordering::AcqRel) {
            if let Err(e) = self.register_callback() {
                return Poll::Ready(Err(e));
            }
        }

        *self.callback_data.waker.lock().unwrap() = Some(cx.waker().clone());

        Poll::Pending
    }
}

/// Async wrapper for CUDA kernel execution
pub struct AsyncKernelExecution {
    stream: Arc<CudaStream>,
}

impl AsyncKernelExecution {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, String> {
        Ok(Self { stream })
    }

    /// Create a future that completes when all operations on the stream finish
    pub fn completion_future(&self) -> CudaFuture {
        CudaFuture::new(self.stream.clone())
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Async memory copy operations
pub struct AsyncMemcpy {
    stream: Arc<CudaStream>,
}

impl AsyncMemcpy {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }

    /// Async copy from host to device
    pub async fn copy_host_to_device(
        &self,
        host_ptr: usize,
        device_ptr: usize,
        bytes: usize,
    ) -> Result<(), String> {
        use crate::ffi::{check_cuda_error, cuMemcpyHtoDAsync_v2};

        unsafe {
            check_cuda_error(cuMemcpyHtoDAsync_v2(
                device_ptr,
                host_ptr as *const c_void,
                bytes,
                self.stream.raw(),
            ))?;
        }

        CudaFuture::new(self.stream.clone()).await
    }

    /// Async copy from device to host
    pub async fn copy_device_to_host(
        &self,
        device_ptr: usize,
        host_ptr: usize,
        bytes: usize,
    ) -> Result<(), String> {
        use crate::ffi::{check_cuda_error, cuMemcpyDtoHAsync_v2};

        unsafe {
            check_cuda_error(cuMemcpyDtoHAsync_v2(
                host_ptr as *mut c_void,
                device_ptr,
                bytes,
                self.stream.raw(),
            ))?;
        }

        CudaFuture::new(self.stream.clone()).await
    }

    /// Async copy from device to device
    pub async fn copy_device_to_device(
        &self,
        src_device_ptr: usize,
        dst_device_ptr: usize,
        bytes: usize,
    ) -> Result<(), String> {
        use crate::ffi::{check_cuda_error, cuMemcpyDtoDAsync_v2};

        unsafe {
            check_cuda_error(cuMemcpyDtoDAsync_v2(
                dst_device_ptr,
                src_device_ptr,
                bytes,
                self.stream.raw(),
            ))?;
        }

        CudaFuture::new(self.stream.clone()).await
    }
}

/// Async execution context for CUDA operations
pub struct AsyncCudaContext {
    streams: Vec<Arc<CudaStream>>,
    current_stream_idx: AtomicUsize,
}

impl AsyncCudaContext {
    /// Create a new async context with the specified number of streams
    pub fn new(num_streams: usize) -> Result<Self, String> {
        let streams: Result<Vec<_>, _> = (0..num_streams)
            .map(|_| CudaStream::new().map(Arc::new))
            .collect();

        Ok(Self {
            streams: streams?,
            current_stream_idx: AtomicUsize::new(0),
        })
    }

    /// Get a stream for async operations (round-robin, lock-free)
    pub fn get_stream(&self) -> Arc<CudaStream> {
        let idx = self.current_stream_idx.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        self.streams[idx].clone()
    }

    /// Get a specific stream by index
    pub fn get_stream_by_index(&self, index: usize) -> Option<Arc<CudaStream>> {
        self.streams.get(index).cloned()
    }

    /// Get the number of streams
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Synchronize all streams
    pub async fn synchronize_all(&self) -> Result<(), String> {
        let futures: Vec<_> = self
            .streams
            .iter()
            .map(|stream| CudaFuture::new(stream.clone()))
            .collect();

        futures::future::try_join_all(futures).await?;
        Ok(())
    }

    /// Execute multiple async operations in parallel across streams
    pub async fn parallel_execute<F, Fut>(&self, operations: Vec<F>) -> Result<Vec<()>, String>
    where
        F: FnOnce(Arc<CudaStream>) -> Fut + Send,
        Fut: Future<Output = Result<(), String>> + Send,
    {
        let futures: Vec<_> = operations
            .into_iter()
            .enumerate()
            .map(|(i, op)| {
                let stream = self.get_stream_by_index(i % self.num_streams()).unwrap();
                op(stream)
            })
            .collect();

        futures::future::try_join_all(futures).await
    }
}