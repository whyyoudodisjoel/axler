/// Module for tensor indexing utilities
/// Calculate strides for a given shape (row-major order)
///
/// # Arguments
/// * `shape` - Shape of the tensor
///
/// # Returns
/// Vector of strides for each dimension
///
/// # Example
/// For shape [2, 3, 4], returns [12, 4, 1]
pub fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert multi-dimensional indices to a linear index expression (for code generation)
///
/// # Arguments
/// * `indices` - Array of index variable names (e.g., ["i0", "i1", "i2"])
/// * `shape` - Shape of the tensor
///
/// # Returns
/// A string expression that computes the linear index
///
/// # Example
/// For shape [2, 3, 4] and indices ["i0", "i1", "i2"]:
/// Returns "(i0*12 + i1*4 + i2)"
pub fn rangeify(indices: &[String], shape: &[usize]) -> String {
    if shape.is_empty() || shape.len() == 1 {
        return if indices.is_empty() {
            "0".to_string()
        } else {
            indices[0].clone()
        };
    }

    let strides = calculate_strides(shape);
    let mut terms = Vec::new();

    for i in 0..shape.len() {
        if strides[i] == 1 {
            terms.push(indices[i].clone());
        } else {
            terms.push(format!("{}*{}", indices[i], strides[i]));
        }
    }

    if terms.len() == 1 {
        terms[0].clone()
    } else {
        format!("({})", terms.join(" + "))
    }
}

/// Decompose a linear index into multi-dimensional indices
///
/// # Arguments
/// * `linear_idx` - The linear index
/// * `shape` - Shape of the tensor
///
/// # Returns
/// Vector of indices for each dimension
///
/// # Example
/// For linear_idx=13 and shape [2, 3, 4], returns [1, 0, 1]
pub fn unrangeify(linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    if shape.len() == 1 {
        return vec![linear_idx];
    }

    let strides = calculate_strides(shape);
    let mut indices = Vec::with_capacity(shape.len());
    let mut remaining = linear_idx;

    for stride in &strides {
        indices.push(remaining / stride);
        remaining %= stride;
    }

    indices
}

/// Convert multi-dimensional indices to linear index (numeric version)
///
/// # Arguments
/// * `indices` - Array of indices for each dimension
/// * `shape` - Shape of the tensor
///
/// # Returns
/// Linear index
///
/// # Example
/// For indices [1, 0, 1] and shape [2, 3, 4], returns 13
pub fn indices_to_linear(indices: &[usize], shape: &[usize]) -> usize {
    let strides = calculate_strides(shape);
    indices
        .iter()
        .zip(&strides)
        .map(|(idx, stride)| idx * stride)
        .sum()
}

/// Get the total number of elements for a given shape
///
/// # Arguments
/// * `shape` - Shape of the tensor
///
/// # Returns
/// Total number of elements
pub fn shape_size(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}
