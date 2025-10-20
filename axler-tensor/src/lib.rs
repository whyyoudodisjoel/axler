use std::ops::{Add, Div, Mul, Sub};

use axler_traits::Device;
use axler_uop::{Buffer, DeviceType, ToDType, UOp};

pub mod async_realize;
pub mod realize;
pub mod tests;

// Re-export async types
pub use async_realize::TensorHandle;

pub struct Tensor {
    pub uop: UOp,
}

// SAFETY: Tensor operations are immutable - they only read from buffers
// and never modify them. The realize() operation creates new buffers
// rather than modifying existing ones.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        let (buf, device) = match &self.uop {
            UOp::Kernel(_, buf, _, device) => (buf, *device),
            UOp::Buffer(buf) if buf.device() == DeviceType::CUDA => (buf, buf.device()),
            _ => return,
        };

        if buf.is_last_reference() {
            match device {
                DeviceType::CPU => {
                    let mut device = axler_cpu::CPU_DEVICE.lock();
                    unsafe {
                        device.deallocate(
                            buf.ptr().f32 as *mut std::ffi::c_void,
                            buf.size(),
                            buf.dtype(),
                        );
                    }
                }
                DeviceType::CUDA => {
                    if let Some(ref mut device) = *axler_cuda::CUDA_DEVICE.lock().unwrap() {
                        unsafe {
                            device.deallocate(
                                buf.ptr().f32 as *mut std::ffi::c_void,
                                buf.size(),
                                buf.dtype(),
                            );
                        }
                    }
                }
            }
        }
    }
}

impl Tensor {
    pub fn from_slice<T>(value: &[T], device: DeviceType) -> Self
    where
        T: ToDType,
    {
        let dtype = T::dtype();
        let ptr = value.as_ptr();
        let len = value.len();
        let buffer_ptr = T::to_buffer_ptr(ptr, len);

        let mut res = Self {
            uop: UOp::Buffer(Buffer::new(
                dtype,
                buffer_ptr,
                axler_uop::DeviceType::CPU, // Host data starts on CPU
                len,
            )),
        };

        if !matches!(device, DeviceType::CPU) {
            res = res.to_device(device);
        }

        res
    }

    pub fn shape(&self) -> Vec<usize> {
        self.uop.shape()
    }

    pub fn dtype(&self) -> axler_uop::DType {
        self.uop.dtype()
    }

    pub fn reshape(&self, sh: &[usize]) -> Self {
        Self {
            uop: UOp::MovementOps(axler_uop::MovementOps::Reshape(
                Box::new(self.uop.clone()),
                sh.to_vec(),
            )),
        }
    }

    pub fn sum(&self, axes: Option<usize>) -> Self {
        Self {
            uop: UOp::ReduceOps(axler_uop::ReduceOps::Sum {
                parent: Box::new(self.uop.clone()),
                axes,
            }),
        }
    }

    pub fn max(&self, axes: Option<usize>) -> Self {
        Self {
            uop: UOp::ReduceOps(axler_uop::ReduceOps::Max {
                parent: Box::new(self.uop.clone()),
                axes,
            }),
        }
    }

    pub fn min(&self, axes: Option<usize>) -> Self {
        Self {
            uop: UOp::ReduceOps(axler_uop::ReduceOps::Min {
                parent: Box::new(self.uop.clone()),
                axes,
            }),
        }
    }

    pub fn mean(&self, axes: Option<usize>) -> Self {
        Self {
            uop: UOp::ReduceOps(axler_uop::ReduceOps::Mean {
                parent: Box::new(self.uop.clone()),
                axes,
            }),
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Add(
                Box::new(self.uop.clone()),
                Box::new(rhs.uop.clone()),
            )),
        }
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Sub(
                Box::new(self.uop.clone()),
                Box::new(rhs.uop.clone()),
            )),
        }
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Mul(
                Box::new(self.uop.clone()),
                Box::new(rhs.uop.clone()),
            )),
        }
    }

    pub fn div(&self, rhs: &Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Div(
                Box::new(self.uop.clone()),
                Box::new(rhs.uop.clone()),
            )),
        }
    }

    pub fn neg(&self, rhs: &Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Neg(
                Box::new(self.uop.clone()),
                Box::new(rhs.uop.clone()),
            )),
        }
    }

    /// Transfer tensor to a specific device
    pub fn to_device(&self, device: DeviceType) -> Self {
        use axler_cpu::get_cpu_device;
        use axler_cuda::get_cuda_device;

        if self.uop.get_target_device() == device {
            return Tensor {
                uop: self.uop.clone(),
            };
        }

        let (source_buffer, shape, is_raw_buffer) = match &self.uop {
            UOp::Buffer(buf) => (buf.clone(), vec![], true),
            UOp::Const(_) => {
                return Tensor {
                    uop: self.uop.clone(),
                }
            }
            UOp::Kernel(_, buf, shape, _) => (buf.clone(), shape.clone(), false),
            _ => {
                let realized = self.realize();
                match &realized.uop {
                    UOp::Kernel(_, buf, shape, _) => (buf.clone(), shape.clone(), false),
                    _ => panic!("Expected Kernel after realization"),
                }
            }
        };

        let (dtype, size) = (source_buffer.dtype(), source_buffer.size());

        let make_buffer = |ptr: *mut std::ffi::c_void, device: DeviceType| {
            axler_uop::Buffer::new(
                dtype,
                unsafe {
                    match dtype {
                        axler_uop::DType::F32 => axler_uop::BufferPtr {
                            f32: std::slice::from_raw_parts(ptr as *const f32, size),
                        },
                        axler_uop::DType::U32 => axler_uop::BufferPtr {
                            u32: std::slice::from_raw_parts(ptr as *const u32, size),
                        },
                        axler_uop::DType::U8 => axler_uop::BufferPtr {
                            u8: std::slice::from_raw_parts(ptr as *const u8, size),
                        },
                    }
                },
                device,
                size,
            )
        };

        let target_buffer = match (source_buffer.device(), device) {
            (DeviceType::CPU, DeviceType::CUDA) => {
                let mut cuda_device = get_cuda_device()
                    .expect("CUDA not initialized")
                    .lock()
                    .unwrap();
                let cuda_device = cuda_device.as_mut().expect("CUDA device not available");
                let device_buf = cuda_device.allocate(size, dtype);
                cuda_device.copy_host_device(
                    source_buffer.get_buffer_ptr(),
                    device_buf,
                    size,
                    dtype,
                );
                make_buffer(device_buf, DeviceType::CUDA)
            }
            (DeviceType::CUDA, DeviceType::CPU) => {
                let cuda_device = get_cuda_device()
                    .expect("CUDA not initialized")
                    .lock()
                    .unwrap();
                let cuda_device = cuda_device.as_ref().expect("CUDA device not available");
                let mut cpu_device = get_cpu_device().lock();
                let host_buf = cpu_device.allocate(size, dtype);
                cuda_device.copy_device_host(source_buffer.get_buffer_ptr(), host_buf, size, dtype);
                make_buffer(host_buf, DeviceType::CPU)
            }
            _ => panic!("Unsupported device transfer"),
        };

        if is_raw_buffer && device == DeviceType::CUDA {
            Tensor {
                uop: UOp::Buffer(target_buffer),
            }
        } else {
            let output_shape = if is_raw_buffer { vec![size] } else { shape };
            Tensor {
                uop: UOp::Kernel(
                    Box::new(self.uop.clone()),
                    target_buffer,
                    output_shape,
                    device,
                ),
            }
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}
