use crate::ffi::CudaStream;
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Callback data structure passed to CUDA stream callback
/// Uses atomic for completion flag and mutex only for the waker
struct CallbackData {
    waker: Mutex<Option<Waker>>,
    completed: AtomicBool,
}

/// A future representing an async CUDA operation
///
/// Uses CUDA stream callbacks (`cuStreamAddCallback`) to signal completion
/// without requiring background threads. This is the most efficient way to
/// wait for CUDA operations in async Rust code.
///
/// # Thread Safety
/// CudaFuture is NOT Send/Sync because CUDA contexts have thread affinity.
/// All CUDA operations must execute on the same thread that created the context.
pub struct CudaFuture {
    stream: Arc<CudaStream>,
    callback_data: Arc<CallbackData>,
    callback_registered: AtomicBool,
}

impl CudaFuture {
    /// Create a new future for operations on the given stream
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

    /// Register the CUDA stream callback (called once during first poll)
    fn register_callback(&self) -> Result<(), String> {
        use crate::ffi::{check_cuda_error, cuStreamAddCallback, CUresult};

        let callback_data_ptr = Arc::into_raw(self.callback_data.clone()) as *mut c_void;

        // CUDA callback that executes when all operations on the stream complete
        extern "C" fn cuda_callback(
            _stream: crate::ffi::raw::CUstream,
            _status: CUresult,
            user_data: *mut c_void,
        ) {
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
                0, // flags (must be 0)
            ))
        }
    }
}

impl Future for CudaFuture {
    type Output = Result<(), String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Fast path: check if already completed (atomic load)
        if self.callback_data.completed.load(Ordering::Acquire) {
            return Poll::Ready(Ok(()));
        }

        // Register callback on first poll (atomic swap ensures it happens once)
        if !self.callback_registered.swap(true, Ordering::AcqRel) {
            if let Err(e) = self.register_callback() {
                return Poll::Ready(Err(e));
            }
        }

        // Store the waker for when CUDA callback fires
        *self.callback_data.waker.lock().unwrap() = Some(cx.waker().clone());

        Poll::Pending
    }
}

/// Async execution context managing multiple CUDA streams
///
/// Provides round-robin load balancing across streams for concurrent execution.
/// All operations use CUDA callbacks for completion notification.
pub struct AsyncCudaContext {
    streams: Vec<Arc<CudaStream>>,
    current_stream_idx: AtomicUsize,
}

impl AsyncCudaContext {
    /// Create a new async context with the specified number of streams
    ///
    /// More streams allow more concurrent operations, but too many can
    /// reduce efficiency. Typical values: 2-8 streams.
    pub fn new(num_streams: usize) -> Result<Self, String> {
        let streams: Result<Vec<_>, _> = (0..num_streams)
            .map(|_| CudaStream::new().map(Arc::new))
            .collect();

        Ok(Self {
            streams: streams?,
            current_stream_idx: AtomicUsize::new(0),
        })
    }

    /// Get a stream for async operations using round-robin (lock-free)
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

    /// Synchronize all streams (wait for all operations to complete)
    pub async fn synchronize_all(&self) -> Result<(), String> {
        let futures: Vec<_> = self
            .streams
            .iter()
            .map(|stream| CudaFuture::new(stream.clone()))
            .collect();

        futures::future::try_join_all(futures).await?;
        Ok(())
    }
}
