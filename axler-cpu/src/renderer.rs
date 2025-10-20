use axler_common::indexing::rangeify;
use axler_traits::Renderer;
use axler_uop::{ALUOps, Buffer, DType, LogicalOps, LoweredUOp, MovementOps, ReduceOps, UOp};
use rustc_hash::FxHashMap;

pub struct CpuRenderer {
    buffer_map: FxHashMap<usize, usize>,
    next_buffer_idx: usize,
}

impl CpuRenderer {
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

    fn render_reduce_op(
        &mut self,
        reduce_op: &ReduceOps<Box<UOp>>,
        lowered_uop: &LoweredUOp,
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
            code.push_str("    // Initialize and reduce all dimensions\n");
            code.push_str(&format!("    buffer_{}[0] = {};\n", output_idx, init_value));

            let input_size: usize = parent_shape.iter().product();
            code.push_str(&format!(
                "    for (size_t i = 0; i < {}; i++) {{\n",
                input_size
            ));

            if parent_shape.len() > 1 {
                code.push_str("        size_t temp_idx = i;\n");
                for dim in (0..parent_shape.len()).rev() {
                    code.push_str(&format!(
                        "        size_t i{} = temp_idx % {};\n",
                        dim, parent_shape[dim]
                    ));
                    if dim > 0 {
                        code.push_str(&format!(
                            "        temp_idx = temp_idx / {};\n",
                            parent_shape[dim]
                        ));
                    }
                }
            }

            let indices: Vec<String> = if parent_shape.len() > 1 {
                (0..parent_shape.len()).map(|i| format!("i{}", i)).collect()
            } else {
                vec!["i".to_string()]
            };

            let input_expr = self.generate_expression(parent, &indices, &parent_shape);

            match reduce_op {
                ReduceOps::Sum { .. } => code.push_str(&format!(
                    "        buffer_{}[0] += {};\n",
                    output_idx, input_expr
                )),
                ReduceOps::Max { .. } => code.push_str(&format!(
                    "        if ({} > buffer_{}[0]) buffer_{}[0] = {};\n",
                    input_expr, output_idx, output_idx, input_expr
                )),
                ReduceOps::Min { .. } => code.push_str(&format!(
                    "        if ({} < buffer_{}[0]) buffer_{}[0] = {};\n",
                    input_expr, output_idx, output_idx, input_expr
                )),
                ReduceOps::Mean { .. } => code.push_str(&format!(
                    "        buffer_{}[0] += {};\n",
                    output_idx, input_expr
                )),
            }

            code.push_str("    }\n");

            if matches!(reduce_op, ReduceOps::Mean { .. }) {
                code.push_str(&format!(
                    "    buffer_{}[0] /= {};\n",
                    output_idx, input_size
                ));
            }
        } else {
            let axis = axes.unwrap();
            code.push_str(&format!("    // Reduce axis {}\n", axis));

            let mut loop_vars = vec![];
            for (dim, &size) in parent_shape.iter().enumerate() {
                if dim != axis {
                    let var = format!("i{}", dim);
                    code.push_str(&format!(
                        "    for (size_t {} = 0; {} < {}; {}++) {{\n",
                        var, var, size, var
                    ));
                    loop_vars.push(var);
                }
            }

            let mut init_indices: Vec<String> = Vec::new();
            for dim in 0..parent_shape.len() {
                if dim != axis {
                    init_indices.push(format!("i{}", dim));
                }
            }

            let init_output_index = if init_indices.is_empty() {
                "0".to_string()
            } else if lowered_uop.output_shape.len() > 1 {
                rangeify(&init_indices, &lowered_uop.output_shape)
            } else {
                init_indices[0].clone()
            };

            code.push_str(&format!(
                "        buffer_{}[{}] = {};\n",
                output_idx, init_output_index, init_value
            ));

            let reduction_var = format!("i{}", axis);
            code.push_str(&format!(
                "        for (size_t {} = 0; {} < {}; {}++) {{\n",
                reduction_var, reduction_var, parent_shape[axis], reduction_var
            ));

            let mut input_indices: Vec<String> = Vec::new();
            for dim in 0..parent_shape.len() {
                input_indices.push(format!("i{}", dim));
            }

            let input_expr = self.generate_expression(parent, &input_indices, &parent_shape);

            let mut output_indices = input_indices.clone();
            output_indices.remove(axis);
            let output_index = if output_indices.is_empty() {
                "0".to_string()
            } else if lowered_uop.output_shape.len() > 1 {
                rangeify(&output_indices, &lowered_uop.output_shape)
            } else {
                output_indices[0].clone()
            };

            match reduce_op {
                ReduceOps::Sum { .. } => code.push_str(&format!(
                    "            buffer_{}[{}] += {};\n",
                    output_idx, output_index, input_expr
                )),
                ReduceOps::Max { .. } => code.push_str(&format!(
                    "            if ({} > buffer_{}[{}]) buffer_{}[{}] = {};\n",
                    input_expr, output_idx, output_index, output_idx, output_index, input_expr
                )),
                ReduceOps::Min { .. } => code.push_str(&format!(
                    "            if ({} < buffer_{}[{}]) buffer_{}[{}] = {};\n",
                    input_expr, output_idx, output_index, output_idx, output_index, input_expr
                )),
                ReduceOps::Mean { .. } => code.push_str(&format!(
                    "            buffer_{}[{}] += {};\n",
                    output_idx, output_index, input_expr
                )),
            }

            code.push_str("        }\n"); // Close reduction loop

            if matches!(reduce_op, ReduceOps::Mean { .. }) {
                code.push_str(&format!(
                    "        buffer_{}[{}] /= {};\n",
                    output_idx, output_index, parent_shape[axis]
                ));
            }

            for _ in loop_vars {
                code.push_str("    }\n");
            }
        }
    }

    fn generate_expression(&mut self, uop: &UOp, indices: &[String], shape: &[usize]) -> String {
        match uop {
            UOp::Buffer(buf) => {
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
            UOp::Const(c) => {
                // consts as literal string
                match c.dtype {
                    DType::F32 => unsafe { format!("{}f", c.value.f32) },
                    DType::U32 => unsafe { format!("{}u", c.value.u32) },
                    DType::U8 => unsafe { format!("{}", c.value.u8) },
                }
            }
            UOp::ALUOps(alu_op) => match alu_op {
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
                    format!("({} % {})", left_expr, right_expr)
                }
                ALUOps::Neg(src, _) => {
                    let src_expr = self.generate_expression(src.as_ref(), indices, shape);
                    format!("(-{})", src_expr)
                }
            },
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
            UOp::MovementOps(mov_op) => {
                match mov_op {
                    MovementOps::Reshape(inner, new_shape) => {
                        // For reshape, use the new shape for indexing
                        self.generate_expression(inner.as_ref(), indices, new_shape)
                    }
                    // TODO: Implement Permute
                    MovementOps::Permute(inner, _) | MovementOps::Pad { parent: inner, .. } => {
                        // self.generate_expression(inner.as_ref(), indices, shape)
                        unimplemented!()
                    }
                }
            }
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
                // TODO: Find a more cleaner pattern without unreachables...
                unreachable!(
                    "Internal error: ReduceOps encountered in generate_expression. \
                     Reduce operations should be handled by render_reduce_op() at the top level."
                )
            }
        }
    }
}

impl Renderer<LoweredUOp> for CpuRenderer {
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

        code.push_str("#include <stddef.h>\n");
        code.push_str("#include <float.h>\n\n");

        code.push_str("// CPU Renderer Generated Code\n");
        code.push_str(&format!(
            "// Output shape: {:?}\n// Output size: {}\n// Output dtype: {:?}\n\n",
            lowered_uop.output_shape, lowered_uop.output_size, lowered_uop.output_dtype
        ));

        self.buffer_map.clear();
        self.next_buffer_idx = 0;

        let buffers = uop.extract_buffers();

        code.push_str("void execute_kernel(void** buffers) {\n");

        for (idx, _) in buffers.iter().enumerate() {
            code.push_str(&format!(
                "    const float* buffer_{} = (const float*)buffers[{}];\n",
                idx, idx
            ));
        }

        let output_idx = buffers.len();
        code.push_str(&format!(
            "    float* buffer_{} = (float*)buffers[{}];\n",
            output_idx, output_idx
        ));
        code.push_str("\n");

        if let UOp::ReduceOps(reduce_op) = uop {
            self.render_reduce_op(reduce_op, &lowered_uop, &mut code, output_idx);
        } else {
            let total_elements = lowered_uop.output_size;
            code.push_str(&format!(
                "    for (size_t i = 0; i < {}; i++) {{\n",
                total_elements
            ));

            if lowered_uop.output_shape.len() > 1 {
                let shape = &lowered_uop.output_shape;
                code.push_str("        // Calculate multi-dimensional indices from linear index\n");
                code.push_str("        size_t temp_idx = i;\n");

                // Generate index variables in reverse order (rightmost dimension varies fastest)
                for dim in (0..shape.len()).rev() {
                    code.push_str(&format!(
                        "        size_t i{} = temp_idx % {};\n",
                        dim, shape[dim]
                    ));
                    if dim > 0 {
                        code.push_str(&format!("        temp_idx = temp_idx / {};\n", shape[dim]));
                    }
                }
                code.push_str("\n");
            }

            // Generate the fused expression
            let indices: Vec<String> = if lowered_uop.output_shape.len() > 1 {
                (0..lowered_uop.output_shape.len())
                    .map(|i| format!("i{}", i))
                    .collect()
            } else {
                vec!["i".to_string()]
            };

            let expression = self.generate_expression(uop, &indices, &lowered_uop.output_shape);

            // Generate output assignment
            let output_index = if lowered_uop.output_shape.len() > 1 {
                rangeify(&indices, &lowered_uop.output_shape)
            } else {
                "i".to_string()
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
