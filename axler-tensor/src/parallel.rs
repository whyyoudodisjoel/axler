use crate::Tensor;
use std::collections::HashSet;

impl<'a> Tensor<'a> {
    pub fn get_buffer_addresses(&self) -> HashSet<usize> {
        let buffers = self.uop.extract_buffers();
        let mut addresses = HashSet::new();

        for buf in buffers {
            let addr = unsafe {
                match buf.dtype {
                    axler_uop::DType::F32 => (*buf.ptr.f32).as_ptr() as usize,
                    axler_uop::DType::U32 => (*buf.ptr.u32).as_ptr() as usize,
                    axler_uop::DType::U8 => (*buf.ptr.u8).as_ptr() as usize,
                }
            };
            addresses.insert(addr);
        }

        addresses
    }

    pub fn has_shared_dependencies(&self, other: &Tensor<'a>) -> bool {
        let self_addrs = self.get_buffer_addresses();
        let other_addrs = other.get_buffer_addresses();

        !self_addrs.is_disjoint(&other_addrs)
    }
}

/// Macro for parallel execution of independent tensor operations
///
/// # Example
/// ```rust
/// use axler_tensor::{Tensor, parallel_execute};
///
/// let a = vec![1.0f32, 2.0, 3.0, 4.0];
/// let b = vec![5.0f32, 6.0, 7.0, 8.0];
/// let c = vec![9.0f32, 10.0, 11.0, 12.0];
/// let d = vec![13.0f32, 14.0, 15.0, 16.0];
///
/// let tensor_a = Tensor::from_slice(&a[..]);
/// let tensor_b = Tensor::from_slice(&b[..]);
/// let tensor_c = Tensor::from_slice(&c[..]);
/// let tensor_d = Tensor::from_slice(&d[..]);
///
/// // Operations must be independent (no shared buffers)
/// let op1 = &tensor_a + &tensor_b;
/// let op2 = &tensor_c * &tensor_d;
///
/// let (result_a, result_b) = parallel_execute!(op1, op2);
/// ```
///
/// # Panics
///
/// The macro will panic if operations share dependencies (use the same input buffers).
#[macro_export]
macro_rules! parallel_execute {
    // Base case: single expression
    ($expr:expr) => {{
        $expr.realize()
    }};

    ($expr1:expr, $expr2:expr) => {{

        let tensor1 = &$expr1;
        let tensor2 = &$expr2;

        assert!(
            !tensor1.has_shared_dependencies(tensor2),
            "Cannot execute tensors in parallel: they share dependencies"
        );

        rayon::join(
            || tensor1.realize(),
            || tensor2.realize()
        )
    }};

    ($expr1:expr, $expr2:expr, $expr3:expr) => {{

        let tensor1 = &$expr1;
        let tensor2 = &$expr2;
        let tensor3 = &$expr3;

        assert!(
            !tensor1.has_shared_dependencies(tensor2),
            "Cannot execute tensors in parallel: tensor1 and tensor2 share dependencies"
        );
        assert!(
            !tensor1.has_shared_dependencies(tensor3),
            "Cannot execute tensors in parallel: tensor1 and tensor3 share dependencies"
        );
        assert!(
            !tensor2.has_shared_dependencies(tensor3),
            "Cannot execute tensors in parallel: tensor2 and tensor3 share dependencies"
        );

        let ((r1, r2), r3) = rayon::join(
            || rayon::join(
                || tensor1.realize(),
                || tensor2.realize()
            ),
            || tensor3.realize()
        );
        (r1, r2, r3)
    }};

    ($expr1:expr, $expr2:expr, $expr3:expr, $expr4:expr) => {{

        let tensor1 = &$expr1;
        let tensor2 = &$expr2;
        let tensor3 = &$expr3;
        let tensor4 = &$expr4;

        let tensors = [tensor1, tensor2, tensor3, tensor4];
        for i in 0..4 {
            for j in (i+1)..4 {
                assert!(
                    !tensors[i].has_shared_dependencies(tensors[j]),
                    "Cannot execute tensors in parallel: tensor{} and tensor{} share dependencies",
                    i+1, j+1
                );
            }
        }

        let ((r1, r2), (r3, r4)) = rayon::join(
            || rayon::join(
                || tensor1.realize(),
                || tensor2.realize()
            ),
            || rayon::join(
                || tensor3.realize(),
                || tensor4.realize()
            )
        );
        (r1, r2, r3, r4)
    }};

    ($($expr:expr),+ $(,)?) => {{
        use $crate::parallel::*;

        let tensors = vec![$($expr),+];

        for i in 0..tensors.len() {
            for j in (i+1)..tensors.len() {
                assert!(
                    !tensors[i].has_shared_dependencies(&tensors[j]),
                    "Cannot execute tensors in parallel: tensor[{}] and tensor[{}] share dependencies",
                    i, j
                );
            }
        }

        use rayon::prelude::*;

        tensors
            .into_par_iter()
            .map(|t| t.realize())
            .collect::<Vec<_>>()
    }};
}

/// Helper function to execute a slice of tensors in parallel
/// Returns realized tensors
pub fn parallel_realize<'a>(tensors: &[&'a Tensor<'a>]) -> Vec<Tensor<'a>> {
    use rayon::prelude::*;

    tensors.par_iter().map(|&tensor| tensor.realize()).collect()
}

/// Helper function to check if a slice of tensors can be executed in parallel
/// Returns true if no dependencies are shared
pub fn can_execute_parallel<'a>(tensors: &[&Tensor<'a>]) -> bool {
    for i in 0..tensors.len() {
        for j in i + 1..tensors.len() {
            if tensors[i].has_shared_dependencies(tensors[j]) {
                return false;
            }
        }
    }
    true
}
