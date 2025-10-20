use axler_common::indexing::rangeify;
use axler_traits::Renderer;
use axler_uop::{ALUOps, Buffer, DType, LogicalOps, LoweredUOp, MovementOps, ReduceOps, UOp};
use rustc_hash::FxHashMap;

pub struct CudaRenderer {
    buffer_map: FxHashMap<usize, usize>,
    next_buffer_idx: usize,
}

impl CudaRenderer {
    pub fn new() -> Self {
        Self {
            buffer_map: FxHashMap::default(),
            next_buffer_idx: 0,
        }
    }

    fn allocate_buffer_idx(&mut self, buffer: Buffer) -> usize {
        let ptr_addr = unsafe {
            match buffer.dtype() {
                DType::F32 => (*buffer.ptr().f32).as_ptr() as usize,
                DType::U32 => (*buffer.ptr().u32).as_ptr() as usize,
                DType::U8 => (*buffer.ptr().u8).as_ptr() as usize,
            }
        };

        if let Some(&idx) = self.buffer_map.get(&ptr_addr) {
            return idx;
        }
        let idx = self.next_buffer_idx;
        self.next_buffer_idx += 1;
        self.buffer_map.insert(ptr_addr, idx);
        idx
    }

    fn generate_expression(&mut self, uop: &UOp, indices: &[String], shape: &[usize]) -> String {
        match uop {
            UOp::Buffer(buf) => {
                let idx = self.allocate_buffer_idx(buf.clone());
                let index_expr = if indices.len() == 1 {
                    // Single index - use flat/linear indexing
                    // Buffers are contiguous in memory, so a single index always works
                    indices[0].clone()
                } else if indices.len() == shape.len() && shape.len() > 1 {
                    // Multi-dimensional indexing - convert indices to linear offset
                    rangeify(indices, shape)
                } else if indices.is_empty() {
                    "0".to_string()
                } else {
                    // Mismatch - shouldn't happen but fall back to first index
                    indices[0].clone()
                };
                format!("buffer_{}[{}]", idx, index_expr)
            }
            UOp::Const(c) => match c.dtype {
                DType::F32 => unsafe { format!("{}f", c.value.f32) },
                DType::U32 => unsafe { format!("{}u", c.value.u32) },
                DType::U8 => unsafe { format!("{}", c.value.u8) },
            },
            UOp::ALUOps(alu_op) => {
                match alu_op {
                    ALUOps::Add(left, right) => {
                        let left_expr = self.generate_expression(left.as_ref(), indices, shape);
                        let right_expr = self.generate_expression(right.as_ref(), indices, shape);
                        format!("({} + {})", left_expr, right_expr)
                    }
                    ALUOps::Sub(left, right) => {
                        let left_expr = self.generate_expression(left.as_ref(), indices, shape);
                        let right_expr = self.generate_expression(right.as_ref(), indices, shape);
                        format!("({} - {})", left_expr, right_expr)
                    }
                    ALUOps::Mul(left, right) => {
                        let left_expr = self.generate_expression(left.as_ref(), indices, shape);
                        let right_expr = self.generate_expression(right.as_ref(), indices, shape);
                        format!("({} * {})", left_expr, right_expr)
                    }
                    ALUOps::Div(left, right) => {
                        let left_expr = self.generate_expression(left.as_ref(), indices, shape);
                        let right_expr = self.generate_expression(right.as_ref(), indices, shape);
                        format!("({} / {})", left_expr, right_expr)
                    }
                    ALUOps::Mod(left, right) => {
                        let left_expr = self.generate_expression(left.as_ref(), indices, shape);
                        let right_expr = self.generate_expression(right.as_ref(), indices, shape);
                        format!("(fmodf({}, {}))", left_expr, right_expr) // Use fmodf for float modulo
                    }
                    ALUOps::Neg(src, _) => {
                        let src_expr = self.generate_expression(src.as_ref(), indices, shape);
                        format!("(-{})", src_expr)
                    }
                }
            }
            UOp::LogicalOps(log_op) => match log_op {
                LogicalOps::And(left, right) => {
                    let left_expr = self.generate_expression(left.as_ref(), indices, shape);
                    let right_expr = self.generate_expression(right.as_ref(), indices, shape);
                    format!("({} & {})", left_expr, right_expr)
                }
                LogicalOps::Or(left, right) => {
                    let left_expr = self.generate_expression(left.as_ref(), indices, shape);
                    let right_expr = self.generate_expression(right.as_ref(), indices, shape);
                    format!("({} | {})", left_expr, right_expr)
                }
                LogicalOps::Xor(left, right) => {
                    let left_expr = self.generate_expression(left.as_ref(), indices, shape);
                    let right_expr = self.generate_expression(right.as_ref(), indices, shape);
                    format!("({} ^ {})", left_expr, right_expr)
                }
                LogicalOps::Not(src, _) => {
                    let src_expr = self.generate_expression(src.as_ref(), indices, shape);
                    format!("(!{})", src_expr)
                }
            },
            UOp::MovementOps(mov_op) => match mov_op {
                MovementOps::Reshape(inner, new_shape) => {
                    // Reshape changes the logical view but not the physical layout
                    // Pass the new_shape down for multi-dimensional indexing
                    // For flat indexing (single index), it doesn't matter
                    self.generate_expression(inner.as_ref(), indices, new_shape)
                }
                MovementOps::Permute(inner, _) | MovementOps::Pad { parent: inner, .. } => {
                    self.generate_expression(inner.as_ref(), indices, shape)
                }
            },
            UOp::Load(parent, _) => self.generate_expression(parent.as_ref(), indices, shape),

            UOp::Kernel(_, buf, _, _) => {
                let idx = self.allocate_buffer_idx(buf.clone());
                let index_expr = if shape.len() > 1 {
                    rangeify(indices, shape)
                } else if indices.is_empty() {
                    "0".to_string()
                } else {
                    indices[0].clone()
                };
                format!("buffer_{}[{}]", idx, index_expr)
            }
            UOp::ReduceOps(_) => {
                unreachable!(
                    "Internal error: ReduceOps encountered in generate_expression. \
                     Reduce operations should be handled by render_reduce_op() at the top level."
                )
            }
        }
    }

    fn render_reduce_op(
        &mut self,
        reduce_op: &ReduceOps<Box<UOp>>,
        code: &mut String,
        output_idx: usize,
    ) {
        let (parent, axes) = match reduce_op {
            ReduceOps::Sum { parent, axes } => (parent.as_ref(), axes),
            ReduceOps::Max { parent, axes } => (parent.as_ref(), axes),
            ReduceOps::Min { parent, axes } => (parent.as_ref(), axes),
            ReduceOps::Mean { parent, axes } => (parent.as_ref(), axes),
        };

        let (parent_shape, _) = parent.calculate_output_info();

        let init_value = match reduce_op {
            ReduceOps::Sum { .. } | ReduceOps::Mean { .. } => "0.0f",
            ReduceOps::Max { .. } => "-FLT_MAX",
            ReduceOps::Min { .. } => "FLT_MAX",
        };

        if axes.is_none() {
            code.push_str("    // Reduce all dimensions (single-threaded)\n");
            code.push_str("    if (idx == 0) {\n");
            code.push_str(&format!(
                "        buffer_{}[0] = {};\n",
                output_idx, init_value
            ));

            let input_size: usize = parent_shape.iter().product();

            code.push_str(&format!(
                "        for (int i = 0; i < {}; i++) {{\n",
                input_size
            ));

            // When reducing all dimensions, we iterate with a flat index 'i'
            // So we pass a flat shape [input_size] instead of the original multidimensional shape
            let indices = vec!["i".to_string()];
            let flat_shape = vec![input_size];
            let input_expr = self.generate_expression(parent, &indices, &flat_shape);

            match reduce_op {
                ReduceOps::Sum { .. } | ReduceOps::Mean { .. } => {
                    code.push_str(&format!(
                        "            buffer_{}[0] += {};\n",
                        output_idx, input_expr
                    ));
                }
                ReduceOps::Max { .. } => {
                    code.push_str(&format!(
                        "            if ({} > buffer_{}[0]) buffer_{}[0] = {};\n",
                        input_expr, output_idx, output_idx, input_expr
                    ));
                }
                ReduceOps::Min { .. } => {
                    code.push_str(&format!(
                        "            if ({} < buffer_{}[0]) buffer_{}[0] = {};\n",
                        input_expr, output_idx, output_idx, input_expr
                    ));
                }
            }

            code.push_str("        }\n");

            if matches!(reduce_op, ReduceOps::Mean { .. }) {
                code.push_str(&format!(
                    "        buffer_{}[0] /= {};\n",
                    output_idx, input_size
                ));
            }

            code.push_str("    }\n");
        } else {
            code.push_str("    // Axis-specific reduction not implemented\n");
        }
    }
}

impl Renderer<LoweredUOp> for CudaRenderer {
    fn lower_if_required(&mut self, uop: &UOp, _buffers: &[Buffer]) -> LoweredUOp {
        let (output_shape, output_dtype) = uop.calculate_output_info();
        let output_size: usize = output_shape.iter().product();

        LoweredUOp {
            output_shape,
            output_size,
            output_dtype,
        }
    }

    fn render(&mut self, lowered_uop: &LoweredUOp, uop: &UOp) -> String {
        let mut code = String::new();
        code.push_str("// CUDA Kernel Generated Code\n");
        code.push_str("#define FLT_MAX 3.402823466e+38F\n");
        code.push_str("#define FLT_MIN 1.175494351e-38F\n\n");

        code.push_str(&format!(
            "// Output shape: {:?}, size: {}, dtype: {:?}\n",
            lowered_uop.output_shape, lowered_uop.output_size, lowered_uop.output_dtype
        ));

        self.buffer_map.clear();
        self.next_buffer_idx = 0;

        let buffers = uop.extract_buffers();

        code.push_str("extern \"C\" __global__ void kernel(");

        for (idx, _) in buffers.iter().enumerate() {
            code.push_str(&format!("const float* buffer_{}, ", idx));
        }

        let output_idx = buffers.len();
        code.push_str(&format!("float* buffer_{}) {{\n", output_idx));

        code.push_str("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");

        if let UOp::ReduceOps(reduce_op) = uop {
            self.render_reduce_op(reduce_op, &mut code, output_idx);
        } else {
            let total_elements = lowered_uop.output_size;
            code.push_str(&format!("    if (idx < {}) {{\n", total_elements));

            if lowered_uop.output_shape.len() > 1 {
                let shape = &lowered_uop.output_shape;
                code.push_str("        int temp_idx = idx;\n");
                for dim in (0..shape.len()).rev() {
                    code.push_str(&format!(
                        "        int i{} = temp_idx % {};\n",
                        dim, shape[dim]
                    ));
                    if dim > 0 {
                        code.push_str(&format!("        temp_idx = temp_idx / {};\n", shape[dim]));
                    }
                }
                code.push_str("\n");
            }

            let indices: Vec<String> = if lowered_uop.output_shape.len() > 1 {
                (0..lowered_uop.output_shape.len())
                    .map(|i| format!("i{}", i))
                    .collect()
            } else {
                vec!["idx".to_string()]
            };

            let expression = self.generate_expression(uop, &indices, &lowered_uop.output_shape);

            let output_index = if lowered_uop.output_shape.len() > 1 {
                rangeify(&indices, &lowered_uop.output_shape)
            } else {
                "idx".to_string()
            };

            code.push_str(&format!(
                "        buffer_{}[{}] = {};\n",
                output_idx, output_index, expression
            ));

            code.push_str("    }\n");
        }

        code.push_str("}\n");
        code
    }
}
