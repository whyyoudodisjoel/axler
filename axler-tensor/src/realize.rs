use axler_cpu::get_cpu_device;
use axler_traits::Device;
use axler_uop::{DeviceType, UOp};
use std::ffi::c_void;

use crate::Tensor;

use axler_cuda::get_cuda_device;

impl Tensor {
    pub fn realize(&self) -> Tensor {
        // Check if already realized
        if matches!(
            self.uop,
            UOp::Kernel(_, _, _, _) | UOp::Buffer(_) | UOp::Const(_)
        ) {
            return Tensor {
                uop: self.uop.clone(),
            };
        }

        let target_device = self.uop.get_target_device();

        let (_kernel_ptr, output_buffer, output_shape, output_dtype, output_size) =
            match target_device {
                DeviceType::CPU => {
                    let mut device = get_cpu_device().lock();
                    self.execute_on_device(&mut *device, target_device)
                }
                DeviceType::CUDA => {
                    if let Some(cuda_device_mutex) = get_cuda_device() {
                        let mut cuda_device_opt = cuda_device_mutex.lock().unwrap();
                        if let Some(ref mut device) = *cuda_device_opt {
                            self.execute_on_device(device, target_device)
                        } else {
                            panic!("CUDA device not available");
                        }
                    } else {
                        panic!("CUDA device not initialized");
                    }
                }
            };

        let buffer = axler_uop::Buffer::new(
            output_dtype,
            unsafe {
                match output_dtype {
                    axler_uop::DType::F32 => axler_uop::BufferPtr {
                        f32: std::slice::from_raw_parts(output_buffer as *const f32, output_size),
                    },
                    axler_uop::DType::U32 => axler_uop::BufferPtr {
                        u32: std::slice::from_raw_parts(output_buffer as *const u32, output_size),
                    },
                    axler_uop::DType::U8 => axler_uop::BufferPtr {
                        u8: std::slice::from_raw_parts(output_buffer as *const u8, output_size),
                    },
                }
            },
            target_device,
            output_size,
        );

        Tensor {
            uop: UOp::Kernel(
                Box::new(self.uop.clone()),
                buffer,
                output_shape,
                target_device,
            ),
        }
    }

    pub fn execute_on_device(
        &self,
        device: &mut dyn Device,
        target_device: DeviceType,
    ) -> (
        *mut c_void,
        *mut c_void,
        Vec<usize>,
        axler_uop::DType,
        usize,
    ) {
        let buffers = self.uop.extract_buffers();

        for buf in &buffers {
            if buf.device() != target_device && buf.device() != DeviceType::CPU {
                panic!(
                    "Cannot execute kernel with buffers on different devices. Found buffer on {:?}, expected {:?}",
                    buf.device(), target_device
                );
            }
        }

        // Use optimized path that caches based on UOp hash
        let (kernel_ptr, output_shape, output_dtype, output_size) = match target_device {
            DeviceType::CPU => {
                // Cast to CpuDevice to use optimized method
                let cpu_device =
                    unsafe { &mut *(device as *mut dyn Device as *mut axler_cpu::CpuDevice) };
                cpu_device.get_or_compile_kernel(&self.uop, &buffers)
            }
            DeviceType::CUDA => {
                // Cast to CudaDevice to use optimized method
                let cuda_device =
                    unsafe { &mut *(device as *mut dyn Device as *mut axler_cuda::CudaDevice) };
                cuda_device.get_or_compile_kernel(&self.uop, &buffers)
            }
        };

        let output_buffer = device.allocate(output_size, output_dtype);

        let mut buffer_ptrs: Vec<*mut c_void> = buffers
            .iter()
            .map(|buf| buf.get_buffer_ptr() as *mut c_void)
            .collect();

        buffer_ptrs.push(output_buffer);

        device.execute(kernel_ptr, buffer_ptrs.clone());

        (
            output_buffer,
            output_buffer,
            output_shape,
            output_dtype,
            output_size,
        )
    }

    pub fn to_vec<T: axler_uop::ToDType + Clone + Default>(&self) -> Vec<T> {
        let realized = self.realize();

        let buffer = match &realized.uop {
            UOp::Kernel(_, buf, _, _) => buf.clone(),
            _ => panic!("Failed to realize tensor"),
        };

        if buffer.dtype() != T::dtype() {
            panic!(
                "Type mismatch: tensor has dtype {:?} but requested type has dtype {:?}",
                buffer.dtype(),
                T::dtype()
            );
        }

        let size = buffer.get_buffer_size();

        match buffer.device() {
            axler_uop::DeviceType::CPU => unsafe {
                let slice = match buffer.dtype() {
                    axler_uop::DType::F32 => {
                        std::slice::from_raw_parts((*buffer.ptr().f32).as_ptr() as *const T, size)
                    }
                    axler_uop::DType::U32 => {
                        std::slice::from_raw_parts((*buffer.ptr().u32).as_ptr() as *const T, size)
                    }
                    axler_uop::DType::U8 => {
                        std::slice::from_raw_parts((*buffer.ptr().u8).as_ptr() as *const T, size)
                    }
                };
                slice.to_vec()
            },
            DeviceType::CUDA => {
                {
                    if let Some(cuda_device_mutex) = axler_cuda::get_cuda_device() {
                        let cuda_device_opt = cuda_device_mutex.lock().unwrap();
                        if let Some(ref cuda_device) = *cuda_device_opt {
                            // Allocate host buffer
                            let mut host_vec = vec![T::default(); size];
                            let host_ptr = host_vec.as_mut_ptr() as *mut c_void;

                            let device_ptr = buffer.get_buffer_ptr();

                            // Use CUDA device's copy_device_host to transfer from CUDA to CPU
                            cuda_device.copy_device_host(
                                device_ptr,
                                host_ptr,
                                size,
                                buffer.dtype(),
                            );
                            return host_vec;
                        }
                    }
                    panic!("CUDA device not available for transfer");
                }
            }
        }
    }
}
