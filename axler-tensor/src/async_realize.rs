use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::Tensor;
use axler_uop::{DeviceType, UOp};

use axler_cuda::get_cuda_device;
use axler_traits::{AsyncDevice, Device};

/// A future handle that resolves to a realized Tensor
/// Users can launch multiple operations and join them with futures::join!()
pub struct TensorHandle {
    /// The main kernel execution future (set after memcpy completes)
    kernel_future: Option<axler_cuda::async_ops::CudaFuture>,
    /// H2D memcpy futures that must complete before kernel launch
    h2d_futures: Vec<axler_cuda::async_ops::CudaFuture>,
    /// Kernel and buffer info for launching after H2D completes
    kernel_info: Option<KernelLaunchInfo>,
    output_buffer: *mut c_void,
    output_shape: Vec<usize>,
    output_dtype: axler_uop::DType,
    output_size: usize,
    target_device: DeviceType,
    parent_uop: UOp,
    temp_buffers: Vec<*mut c_void>,
    temp_sizes: Vec<usize>,
    temp_dtypes: Vec<axler_uop::DType>,
}

struct KernelLaunchInfo {
    kernel_handle: axler_traits::KernelHandle,
    buffer_ptrs: Vec<*mut c_void>,
}

impl Future for TensorHandle {
    type Output = Result<Tensor, String>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // First, poll all H2D memcpy futures
        if !self.h2d_futures.is_empty() {
            let mut all_ready = true;
            for fut in &mut self.h2d_futures {
                match Pin::new(fut).poll(cx) {
                    Poll::Ready(Ok(())) => continue,
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                    Poll::Pending => {
                        all_ready = false;
                    }
                }
            }

            if !all_ready {
                return Poll::Pending;
            }

            // All H2D transfers complete, launch the kernel
            self.h2d_futures.clear();

            if let Some(launch_info) = self.kernel_info.take() {
                if let Some(cuda_device_mutex) = get_cuda_device() {
                    let mut cuda_device_opt = cuda_device_mutex.lock().unwrap();
                    if let Some(ref mut device) = *cuda_device_opt {
                        match device
                            .execute_async(launch_info.kernel_handle, launch_info.buffer_ptrs)
                        {
                            Ok(fut) => {
                                self.kernel_future = Some(fut);
                            }
                            Err(e) => return Poll::Ready(Err(e)),
                        }
                    } else {
                        return Poll::Ready(Err("CUDA device not available".to_string()));
                    }
                } else {
                    return Poll::Ready(Err("CUDA device not initialized".to_string()));
                }
            }
        }

        // Now poll the kernel execution future
        if let Some(ref mut kernel_fut) = self.kernel_future {
            match Pin::new(kernel_fut).poll(cx) {
                Poll::Ready(Ok(())) => {
                    if let Some(cuda_device_mutex) = get_cuda_device() {
                        let mut cuda_device_opt = cuda_device_mutex.lock().unwrap();
                        if let Some(ref mut device) = *cuda_device_opt {
                            for i in 0..self.temp_buffers.len() {
                                device.deallocate_async(
                                    self.temp_buffers[i],
                                    self.temp_sizes[i],
                                    self.temp_dtypes[i],
                                );
                            }
                        }
                    }

                    // Construct the output buffer
                    let buffer = axler_uop::Buffer::new(
                        self.output_dtype,
                        unsafe {
                            match self.output_dtype {
                                axler_uop::DType::F32 => axler_uop::BufferPtr {
                                    f32: std::slice::from_raw_parts(
                                        self.output_buffer as *const f32,
                                        self.output_size,
                                    ),
                                },
                                axler_uop::DType::U32 => axler_uop::BufferPtr {
                                    u32: std::slice::from_raw_parts(
                                        self.output_buffer as *const u32,
                                        self.output_size,
                                    ),
                                },
                                axler_uop::DType::U8 => axler_uop::BufferPtr {
                                    u8: std::slice::from_raw_parts(
                                        self.output_buffer as *const u8,
                                        self.output_size,
                                    ),
                                },
                            }
                        },
                        self.target_device,
                        self.output_size,
                    );

                    Poll::Ready(Ok(Tensor {
                        uop: UOp::Kernel(
                            Box::new(self.parent_uop.clone()),
                            buffer,
                            self.output_shape.clone(),
                            self.target_device,
                        ),
                    }))
                }
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Pending
        }
    }
}

impl Tensor {
    /// Launch async realization and return a future handle
    /// Use this to launch multiple operations and join them with futures::join!()
    ///
    /// # Example
    /// ```ignore
    /// let handle1 = tensor1.spawn_realize()?;
    /// let handle2 = tensor2.spawn_realize()?;
    /// let (result1, result2) = futures::join!(handle1, handle2);
    /// ```
    pub fn spawn_realize(&self) -> Result<TensorHandle, String> {
        // Check if already realized
        if matches!(
            self.uop,
            UOp::Kernel(_, _, _, _) | UOp::Buffer(_) | UOp::Const(_)
        ) {
            return Err("Tensor is already realized".to_string());
        }

        let target_device = self.uop.get_target_device();

        if target_device != DeviceType::CUDA {
            return Err("spawn_realize only supports CUDA. Use realize() for CPU.".to_string());
        }

        let buffers = self.uop.extract_buffers();

        for buf in &buffers {
            if buf.device() != target_device && buf.device() != DeviceType::CPU {
                return Err(format!(
                    "Cannot execute kernel with buffers on different devices. Found buffer on {:?}, expected {:?}",
                    buf.device(), target_device
                ));
            }
        }

        if let Some(cuda_device_mutex) = get_cuda_device() {
            let mut cuda_device_opt = cuda_device_mutex.lock().unwrap();
            if let Some(ref mut device) = *cuda_device_opt {
                // Get or compile kernel
                let (kernel_ptr, output_shape, output_dtype, output_size) =
                    device.get_or_compile_kernel(&self.uop, &buffers);

                let output_buffer = device.allocate_async(output_size, output_dtype);

                let mut temp_buffers = Vec::new();
                let mut temp_sizes = Vec::new();
                let mut temp_dtypes = Vec::new();
                let mut h2d_futures = Vec::new();

                // Prepare buffer pointers and launch async H2D transfers
                let mut buffer_ptrs: Vec<*mut c_void> = Vec::new();

                for buf in &buffers {
                    if buf.device() == DeviceType::CUDA {
                        buffer_ptrs.push(buf.get_buffer_ptr() as *mut c_void);
                    } else {
                        // Allocate device buffer
                        let size = buf.get_buffer_size();
                        let device_buf = device.allocate_async(size, buf.dtype());
                        let host_ptr = buf.get_buffer_ptr();

                        // Launch async H2D copy
                        let h2d_future = device.copy_host_device_async(
                            host_ptr,
                            device_buf,
                            size,
                            buf.dtype(),
                        )?;

                        h2d_futures.push(h2d_future);

                        // Track for later deallocation
                        temp_buffers.push(device_buf);
                        temp_sizes.push(size);
                        temp_dtypes.push(buf.dtype());

                        buffer_ptrs.push(device_buf);
                    }
                }

                buffer_ptrs.push(output_buffer);

                // Return the handle with H2D futures
                // Kernel will be launched after H2D completes
                Ok(TensorHandle {
                    kernel_future: None,
                    h2d_futures,
                    kernel_info: Some(KernelLaunchInfo {
                        kernel_handle: kernel_ptr,
                        buffer_ptrs,
                    }),
                    output_buffer,
                    output_shape,
                    output_dtype,
                    output_size,
                    target_device,
                    parent_uop: self.uop.clone(),
                    temp_buffers,
                    temp_sizes,
                    temp_dtypes,
                })
            } else {
                Err("CUDA device not available".to_string())
            }
        } else {
            Err("CUDA device not initialized".to_string())
        }
    }
}
