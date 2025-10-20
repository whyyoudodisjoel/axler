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
    kernel_future: Option<axler_cuda::async_ops::CudaFuture>,
    output_buffer: *mut c_void,
    output_shape: Vec<usize>,
    output_dtype: axler_uop::DType,
    output_size: usize,
    target_device: DeviceType,
    parent_uop: UOp,
}

impl Future for TensorHandle {
    type Output = Result<Tensor, String>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Poll the kernel execution future directly (no H2D transfers)
        if let Some(ref mut kernel_fut) = self.kernel_future {
            match Pin::new(kernel_fut).poll(cx) {
                Poll::Ready(Ok(())) => {
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

                let mut buffer_ptrs: Vec<*mut c_void> = buffers
                    .iter()
                    .map(|buf| buf.get_buffer_ptr() as *mut c_void)
                    .collect();

                buffer_ptrs.push(output_buffer);

                let kernel_future = device.execute_async(kernel_ptr, buffer_ptrs)?;

                Ok(TensorHandle {
                    kernel_future: Some(kernel_future),
                    output_buffer,
                    output_shape,
                    output_dtype,
                    output_size,
                    target_device,
                    parent_uop: self.uop.clone(),
                })
            } else {
                Err("CUDA device not available".to_string())
            }
        } else {
            Err("CUDA device not initialized".to_string())
        }
    }
}
