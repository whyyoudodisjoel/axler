use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct LoweredUOp {
    pub output_shape: Vec<usize>,
    pub output_size: usize,
    pub output_dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    U32,
    U8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    CPU,
    CUDA,
    // Future: Metal, etc.
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union ConstValue {
    pub f32: f32,
    pub u32: u32,
    pub u8: u8,
}

impl Debug for ConstValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ConstValue")
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union BufferPtr {
    pub f32: *const [f32],
    pub u32: *const [u32],
    pub u8: *const [u8],
}

impl Debug for BufferPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BufferPtr")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Buffer {
    pub dtype: DType,
    pub ptr: BufferPtr,
    pub device: DeviceType,
    pub size: usize, // Number of elements (not bytes)
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.device == other.device && self.ptr_equals(&other.ptr)
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.device.hash(state);
        unsafe {
            match self.dtype {
                DType::F32 => (self.ptr.f32 as *const f32).hash(state),
                DType::U32 => (self.ptr.u32 as *const u32).hash(state),
                DType::U8 => (self.ptr.u8 as *const u8).hash(state),
            }
        }
    }
}

impl Buffer {
    fn ptr_equals(&self, other: &BufferPtr) -> bool {
        unsafe {
            match self.dtype {
                DType::F32 => self.ptr.f32 as *const f32 == other.f32 as *const f32,
                DType::U32 => self.ptr.u32 as *const u32 == other.u32 as *const u32,
                DType::U8 => self.ptr.u8 as *const u8 == other.u8 as *const u8,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Const {
    pub dtype: DType,
    pub value: ConstValue,
}

#[derive(Debug, Clone)]
pub enum UOp {
    Const(Const),
    Buffer(Buffer),

    // ALU Ops
    ALUOps(ALUOps<Box<Self>>),

    // Logical Ops
    LogicalOps(LogicalOps<Box<Self>>),

    // Movement Ops
    MovementOps(MovementOps<Box<Self>>),

    ReduceOps(ReduceOps<Box<Self>>),

    // Load operation - transfers data from one device to another
    // Contains the source UOp and target device
    Load(Box<Self>, DeviceType),

    // Kernel Boundary - stores the AST (for gradients), realized buffer, output shape, and device
    Kernel(Box<Self>, Buffer, Vec<usize>, DeviceType),
}

#[derive(Debug, Clone)]
pub enum ReduceOps<T> {
    Sum {
        parent: T,
        axes: Option<usize>, // None => Apply to all elems
    },
    Max {
        parent: T,
        axes: Option<usize>, // None => Apply to all elems
    },
    Min {
        parent: T,
        axes: Option<usize>, // None => Apply to all elems
    },
    Mean {
        parent: T,
        axes: Option<usize>, // None => Apply to all elems
    },
}

#[derive(Debug, Clone)]
pub enum ALUOps<T> {
    Add(T, T),
    Sub(T, T),
    Mul(T, T),
    Div(T, T),
    Neg(T, T),
    Mod(T, T),
}

#[derive(Debug, Clone)]
pub enum LogicalOps<T> {
    And(T, T),
    Or(T, T),
    Not(T, T),
    Xor(T, T),
}

#[derive(Debug, Clone)]
pub enum MovementOps<T> {
    Reshape(T, Vec<usize>), // all reshapes are views
    Permute(T, Vec<usize>),
    Pad { parent: T, pad: Vec<usize>, fill: T },
}

pub trait ToDType {
    fn dtype() -> DType;
    fn to_buffer_ptr(ptr: *const Self, len: usize) -> BufferPtr;
}

impl ToDType for f32 {
    fn dtype() -> DType {
        DType::F32
    }

    fn to_buffer_ptr(ptr: *const Self, len: usize) -> BufferPtr {
        BufferPtr {
            f32: std::ptr::slice_from_raw_parts(ptr, len),
        }
    }
}

impl ToDType for u32 {
    fn dtype() -> DType {
        DType::U32
    }

    fn to_buffer_ptr(ptr: *const Self, len: usize) -> BufferPtr {
        BufferPtr {
            u32: std::ptr::slice_from_raw_parts(ptr, len),
        }
    }
}

impl ToDType for u8 {
    fn dtype() -> DType {
        DType::U8
    }

    fn to_buffer_ptr(ptr: *const Self, len: usize) -> BufferPtr {
        BufferPtr {
            u8: std::ptr::slice_from_raw_parts(ptr, len),
        }
    }
}

use rustc_hash::FxHashSet;
use std::fmt::Debug;

impl UOp {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            UOp::Buffer(buf) => {
                let size = unsafe {
                    match buf.dtype {
                        DType::F32 => (&(*buf.ptr.f32)).len(),
                        DType::U32 => (&(*buf.ptr.u32)).len(),
                        DType::U8 => (&(*buf.ptr.u8)).len(),
                    }
                };
                vec![size]
            }
            UOp::Const(_) => vec![1],
            UOp::ALUOps(op) => match op {
                ALUOps::Add(left, _)
                | ALUOps::Sub(left, _)
                | ALUOps::Mul(left, _)
                | ALUOps::Div(left, _)
                | ALUOps::Mod(left, _) => left.as_ref().shape(),
                ALUOps::Neg(src, _) => src.as_ref().shape(),
            },
            UOp::LogicalOps(op) => match op {
                LogicalOps::And(left, _) | LogicalOps::Or(left, _) | LogicalOps::Xor(left, _) => {
                    left.as_ref().shape()
                }
                LogicalOps::Not(src, _) => src.as_ref().shape(),
            },
            UOp::MovementOps(op) => match op {
                MovementOps::Reshape(_, new_shape) => new_shape.clone(),
                MovementOps::Permute(inner, perm) => {
                    let shape = inner.as_ref().shape();
                    // Permute rearranges dimensions according to perm
                    let mut new_shape = vec![0; shape.len()];
                    for (i, &p) in perm.iter().enumerate() {
                        if p < shape.len() {
                            new_shape[i] = shape[p];
                        }
                    }
                    new_shape
                }
                MovementOps::Pad { parent, pad, .. } => {
                    let shape = parent.as_ref().shape();
                    // Padding increases shape by pad amount on each dimension
                    let mut padded_shape = shape.clone();
                    for (i, &p) in pad.iter().enumerate().take(padded_shape.len()) {
                        padded_shape[i] += p * 2; // Assuming symmetric padding
                    }
                    padded_shape
                }
            },
            UOp::Load(parent, _) => parent.as_ref().shape(),
            UOp::Kernel(_, _, shape, _) => shape.clone(),
            UOp::ReduceOps(op) => match op {
                ReduceOps::Sum { parent, axes }
                | ReduceOps::Max { parent, axes }
                | ReduceOps::Min { parent, axes }
                | ReduceOps::Mean { parent, axes } => {
                    let mut shape = parent.as_ref().shape();

                    if let Some(axis) = axes {
                        if *axis < shape.len() {
                            shape.remove(*axis);
                            if shape.is_empty() {
                                shape = vec![1];
                            }
                        }
                    } else {
                        shape = vec![1];
                    }

                    shape
                }
            },
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            UOp::Buffer(buf) => buf.dtype,
            UOp::Const(c) => c.dtype,
            UOp::ALUOps(op) => match op {
                ALUOps::Add(left, _)
                | ALUOps::Sub(left, _)
                | ALUOps::Mul(left, _)
                | ALUOps::Div(left, _)
                | ALUOps::Mod(left, _) => left.as_ref().dtype(),
                ALUOps::Neg(src, _) => src.as_ref().dtype(),
            },
            UOp::LogicalOps(op) => match op {
                LogicalOps::And(left, _) | LogicalOps::Or(left, _) | LogicalOps::Xor(left, _) => {
                    left.as_ref().dtype()
                }
                LogicalOps::Not(src, _) => src.as_ref().dtype(),
            },
            UOp::MovementOps(op) => match op {
                MovementOps::Reshape(inner, _) | MovementOps::Permute(inner, _) => {
                    inner.as_ref().dtype()
                }
                MovementOps::Pad { parent, .. } => parent.as_ref().dtype(),
            },
            UOp::Load(parent, _) => parent.as_ref().dtype(),
            UOp::Kernel(_, buf, _, _) => buf.dtype,
            UOp::ReduceOps(op) => match op {
                ReduceOps::Sum { parent, .. }
                | ReduceOps::Max { parent, .. }
                | ReduceOps::Min { parent, .. }
                | ReduceOps::Mean { parent, .. } => parent.as_ref().dtype(),
            },
        }
    }

    pub fn calculate_output_info(&self) -> (Vec<usize>, DType) {
        (self.shape(), self.dtype())
    }

    pub fn toposort<'a>(&'a self) -> Vec<&'a UOp> {
        let mut visited = FxHashSet::default();
        let mut stack = Vec::new();

        self.dfs_postorder(&mut visited, &mut stack);

        stack.reverse();
        stack
    }

    pub fn extract_buffers(&self) -> Vec<Buffer> {
        let mut buffers = Vec::new();
        // Use FxHashSet which is much faster than std HashSet for pointer keys
        let mut visited = FxHashSet::default();
        self.extract_buffers_recursive(&mut buffers, &mut visited);
        buffers
    }

    fn extract_buffers_recursive(
        &self,
        buffers: &mut Vec<Buffer>,
        visited: &mut FxHashSet<*const UOp>,
    ) {
        let self_ptr = self as *const UOp;

        if visited.contains(&self_ptr) {
            return;
        }

        visited.insert(self_ptr);

        match self {
            UOp::Buffer(buf) => {
                buffers.push(*buf);
            }
            UOp::Const(_) => {
                // Leaf node, no buffers
            }
            UOp::ALUOps(op) => match op {
                ALUOps::Add(a, b)
                | ALUOps::Sub(a, b)
                | ALUOps::Mul(a, b)
                | ALUOps::Div(a, b)
                | ALUOps::Neg(a, b)
                | ALUOps::Mod(a, b) => {
                    a.as_ref().extract_buffers_recursive(buffers, visited);
                    b.as_ref().extract_buffers_recursive(buffers, visited);
                }
            },
            UOp::LogicalOps(op) => match op {
                LogicalOps::And(a, b)
                | LogicalOps::Or(a, b)
                | LogicalOps::Not(a, b)
                | LogicalOps::Xor(a, b) => {
                    a.as_ref().extract_buffers_recursive(buffers, visited);
                    b.as_ref().extract_buffers_recursive(buffers, visited);
                }
            },
            UOp::MovementOps(op) => match op {
                MovementOps::Reshape(a, _) | MovementOps::Permute(a, _) => {
                    a.as_ref().extract_buffers_recursive(buffers, visited)
                }
                MovementOps::Pad { parent, fill, .. } => {
                    parent.as_ref().extract_buffers_recursive(buffers, visited);
                    fill.as_ref().extract_buffers_recursive(buffers, visited);
                }
            },
            UOp::ReduceOps(op) => match op {
                ReduceOps::Max { parent, .. }
                | ReduceOps::Mean { parent, .. }
                | ReduceOps::Min { parent, .. }
                | ReduceOps::Sum { parent, .. } => {
                    parent.as_ref().extract_buffers_recursive(buffers, visited);
                }
            },
            UOp::Load(parent, _) => {
                // Load transfers data between devices
                parent.as_ref().extract_buffers_recursive(buffers, visited);
            }
            UOp::Kernel(_, buf, _, _) => {
                buffers.push(*buf);
                // Don't traverse the AST part - it's already realized
            }
        }
    }

    fn dfs_postorder<'a>(&'a self, visited: &mut FxHashSet<*const UOp>, stack: &mut Vec<&'a UOp>) {
        let self_ptr = self as *const UOp;

        if visited.contains(&self_ptr) {
            return;
        }

        visited.insert(self_ptr);

        match self {
            UOp::Const { .. } | UOp::Buffer { .. } => {
                // Leaf nodes, no dependencies
            }
            UOp::ALUOps(op) => match op {
                ALUOps::Add(a, b)
                | ALUOps::Sub(a, b)
                | ALUOps::Mul(a, b)
                | ALUOps::Div(a, b)
                | ALUOps::Neg(a, b)
                | ALUOps::Mod(a, b) => {
                    a.as_ref().dfs_postorder(visited, stack);
                    b.as_ref().dfs_postorder(visited, stack);
                }
            },
            UOp::LogicalOps(op) => match op {
                LogicalOps::And(a, b)
                | LogicalOps::Or(a, b)
                | LogicalOps::Not(a, b)
                | LogicalOps::Xor(a, b) => {
                    a.as_ref().dfs_postorder(visited, stack);
                    b.as_ref().dfs_postorder(visited, stack);
                }
            },
            UOp::MovementOps(op) => match op {
                MovementOps::Reshape(a, _) | MovementOps::Permute(a, _) => {
                    a.as_ref().dfs_postorder(visited, stack)
                }
                MovementOps::Pad { parent, fill, .. } => {
                    parent.as_ref().dfs_postorder(visited, stack);
                    fill.as_ref().dfs_postorder(visited, stack);
                }
            },
            UOp::ReduceOps(op) => match op {
                ReduceOps::Max { parent, .. }
                | ReduceOps::Mean { parent, .. }
                | ReduceOps::Min { parent, .. }
                | ReduceOps::Sum { parent, .. } => {
                    parent.as_ref().dfs_postorder(visited, stack);
                }
            },
            UOp::Load(parent, _) => {
                parent.as_ref().dfs_postorder(visited, stack);
            }
            UOp::Kernel(_, _, _, _) => {
                // Kernel boundary - don't traverse the AST, it's already realized
            }
        }

        stack.push(self);
    }

    /// Compute a hash for this UOp tree for caching purposes
    pub fn compute_hash(&self) -> u64 {
        use rustc_hash::FxHasher;

        fn hash_uop_recursive<H: Hasher>(uop: &UOp, hasher: &mut H) {
            // Hash the discriminant (which variant)
            std::mem::discriminant(uop).hash(hasher);

            match uop {
                UOp::Const(val) => {
                    val.dtype.hash(hasher);
                    unsafe {
                        match val.dtype {
                            DType::F32 => val.value.f32.to_bits().hash(hasher),
                            DType::U32 => val.value.u32.hash(hasher),
                            DType::U8 => val.value.u8.hash(hasher),
                        }
                    }
                }
                UOp::Load(shape, device) => {
                    hash_uop_recursive(shape, hasher);
                    device.hash(hasher);
                }
                UOp::Buffer(buf) => {
                    buf.dtype.hash(hasher);
                    buf.device.hash(hasher);
                    // Hash the buffer size, not the pointer
                    unsafe {
                        match buf.dtype {
                            DType::F32 => (&*buf.ptr.f32).len().hash(hasher),
                            DType::U32 => (&*buf.ptr.u32).len().hash(hasher),
                            DType::U8 => (&*buf.ptr.u8).len().hash(hasher),
                        }
                    }
                }
                UOp::ALUOps(op) => match op {
                    ALUOps::Add(l, r) => {
                        0u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                    ALUOps::Sub(l, r) => {
                        1u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                    ALUOps::Mul(l, r) => {
                        2u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                    ALUOps::Div(l, r) => {
                        3u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                    ALUOps::Mod(l, r) => {
                        4u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                    ALUOps::Neg(l, r) => {
                        5u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                },
                UOp::LogicalOps(op) => match op {
                    LogicalOps::And(l, r) => {
                        0u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                    LogicalOps::Or(l, r) => {
                        1u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                    LogicalOps::Xor(l, r) => {
                        2u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                    LogicalOps::Not(l, r) => {
                        3u8.hash(hasher);
                        hash_uop_recursive(l, hasher);
                        hash_uop_recursive(r, hasher);
                    }
                },
                UOp::MovementOps(op) => match op {
                    MovementOps::Reshape(inner, shape) => {
                        0u8.hash(hasher);
                        hash_uop_recursive(inner, hasher);
                        shape.hash(hasher);
                    }
                    MovementOps::Permute(inner, perm) => {
                        1u8.hash(hasher);
                        hash_uop_recursive(inner, hasher);
                        perm.hash(hasher);
                    }
                    MovementOps::Pad { parent, pad, fill } => {
                        2u8.hash(hasher);
                        hash_uop_recursive(parent, hasher);
                        pad.hash(hasher);
                        hash_uop_recursive(fill, hasher);
                    }
                },
                UOp::ReduceOps(op) => match op {
                    ReduceOps::Sum { parent, axes } => {
                        0u8.hash(hasher);
                        hash_uop_recursive(parent, hasher);
                        axes.hash(hasher);
                    }
                    ReduceOps::Max { parent, axes } => {
                        1u8.hash(hasher);
                        hash_uop_recursive(parent, hasher);
                        axes.hash(hasher);
                    }
                    ReduceOps::Min { parent, axes } => {
                        2u8.hash(hasher);
                        hash_uop_recursive(parent, hasher);
                        axes.hash(hasher);
                    }
                    ReduceOps::Mean { parent, axes } => {
                        3u8.hash(hasher);
                        hash_uop_recursive(parent, hasher);
                        axes.hash(hasher);
                    }
                },
                UOp::Kernel(inner, buf, shape, device) => {
                    hash_uop_recursive(inner, hasher);
                    buf.dtype.hash(hasher);
                    buf.device.hash(hasher);
                    shape.hash(hasher);
                    device.hash(hasher);
                }
            }
        }

        let mut hasher = FxHasher::default();
        hash_uop_recursive(self, &mut hasher);
        hasher.finish()
    }
}
