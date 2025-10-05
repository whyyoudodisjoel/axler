use std::ops::{Add, Div, Mul, Sub};

use axler_traits::Device;
use axler_uop::{Buffer, DeviceType, ToDType, UOp};

pub mod realize;
pub mod tests;

#[derive(Clone)]
pub struct Tensor<'a> {
    pub uop: UOp<'a>,
}

// SAFETY: Tensor operations are immutable - they only read from buffers
// and never modify them. The realize() operation creates new buffers
// rather than modifying existing ones.
unsafe impl<'a> Send for Tensor<'a> {}
unsafe impl<'a> Sync for Tensor<'a> {}

impl<'a> Drop for Tensor<'a> {
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
                    #[cfg(feature = "cuda")]
                    {
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
}

impl<'a> Tensor<'a> {
    pub fn from_slice<T>(value: &'a [T]) -> Self
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

    pub fn reshape(&'a self, sh: &[usize]) -> Self {
        Self {
            uop: UOp::MovementOps(axler_uop::MovementOps::Reshape(&self.uop, sh.to_vec())),
        }
    }

    pub fn sum(&'a self, axes: Option<usize>) -> Self {
        Self {
            uop: UOp::ReduceOps(axler_uop::ReduceOps::Sum {
                parent: &self.uop,
                axes,
            }),
        }
    }

    pub fn max(&'a self, axes: Option<usize>) -> Self {
        Self {
            uop: UOp::ReduceOps(axler_uop::ReduceOps::Max {
                parent: &self.uop,
                axes,
            }),
        }
    }

    pub fn min(&'a self, axes: Option<usize>) -> Self {
        Self {
            uop: UOp::ReduceOps(axler_uop::ReduceOps::Min {
                parent: &self.uop,
                axes,
            }),
        }
    }

    pub fn mean(&'a self, axes: Option<usize>) -> Self {
        Self {
            uop: UOp::ReduceOps(axler_uop::ReduceOps::Mean {
                parent: &self.uop,
                axes,
            }),
        }
    }

    pub fn add(&'a self, rhs: &'a Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Add(&self.uop, &rhs.uop)),
        }
    }

    pub fn sub(&'a self, rhs: &'a Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Sub(&self.uop, &rhs.uop)),
        }
    }

    pub fn mul(&'a self, rhs: &'a Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Mul(&self.uop, &rhs.uop)),
        }
    }

    pub fn div(&'a self, rhs: &'a Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Div(&self.uop, &rhs.uop)),
        }
    }

    pub fn neg(&'a self, rhs: &'a Self) -> Self {
        Tensor {
            uop: UOp::ALUOps(axler_uop::ALUOps::Neg(&self.uop, &rhs.uop)),
        }
    }

    /// Transfer tensor to a specific device
    /// This creates a Load operation that will transfer data when realized
    pub fn to_device(&'a self, device: DeviceType) -> Self {
        Tensor {
            uop: UOp::Load(&self.uop, device),
        }
    }
}

impl<'a> Add for &'a Tensor<'a> {
    type Output = Tensor<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl<'a> Sub for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl<'a> Mul for &'a Tensor<'a> {
    type Output = Tensor<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl<'a> Div for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}
