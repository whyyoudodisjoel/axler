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
        // Free any realized buffers when the tensor is dropped
        if let UOp::Kernel(_, buf, _, device) = &self.uop {
            // Get the appropriate device and deallocate
            match device {
                DeviceType::CPU => {
                    let mut device = axler_cpu::CPU_DEVICE.lock();
                    unsafe {
                        device.deallocate(
                            buf.ptr.f32 as *mut std::ffi::c_void,
                            buf.size,
                            buf.dtype,
                        );
                    }
                }
                DeviceType::CUDA => {
                    if let Some(ref mut device) = *axler_cuda::CUDA_DEVICE.lock().unwrap() {
                        unsafe {
                            device.deallocate(
                                buf.ptr.f32 as *mut std::ffi::c_void,
                                buf.size,
                                buf.dtype,
                            );
                        }
                    }
                }
            }
        }
    }
}

impl Tensor {
    pub fn from_slice<T>(value: &[T]) -> Self
    where
        T: ToDType,
    {
        let dtype = T::dtype();
        let ptr = value.as_ptr();
        let len = value.len();
        let buffer_ptr = T::to_buffer_ptr(ptr, len);

        Self {
            uop: UOp::Buffer(Buffer {
                dtype,
                ptr: buffer_ptr,
                device: axler_uop::DeviceType::CPU, // Host data starts on CPU
                size: len,
            }),
        }
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
    /// This realizes the tensor first (if needed), then creates a Load operation
    /// that will copy data to the target device when the Load is realized
    pub fn to_device(&self, device: DeviceType) -> Self {
        // If already on target device, return as-is
        let current_device = self.uop.get_target_device();
        if current_device == device {
            return Tensor {
                uop: self.uop.clone(),
            };
        }

        // Realize the parent computation first if it's not already realized
        let realized_parent = match &self.uop {
            UOp::Kernel(_, _, _, _) | UOp::Buffer(_) | UOp::Const(_) => {
                // Already realized, use as-is
                Tensor {
                    uop: self.uop.clone(),
                }
            }
            _ => {
                // Unrealized - realize on current device first
                self.realize()
            }
        };

        // Now create Load with the realized parent
        Tensor {
            uop: UOp::Load(Box::new(realized_parent.uop.clone()), device),
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
