#[allow(unused_imports)]
use crate::Tensor;
#[allow(unused_imports)]
use axler_uop::DeviceType;

#[test]
fn test_realize() {
    let buf = &[1., 2., 3., 4.];
    let tensor = Tensor::from_slice(&buf[..]);

    let res = &tensor + &tensor;

    let output = res.to_vec::<f32>();
    assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
    println!("Test passed! Result: {:?}", output);

    // Run the same computation again to test in-memory cache
    let res2 = &tensor + &tensor;
    let output2 = res2.to_vec::<f32>();
    assert_eq!(output2, vec![2.0, 4.0, 6.0, 8.0]);
    println!("Second run passed! Result: {:?}", output2);
}

#[test]
fn test_multi_dim() {
    let buf = &[1., 2., 3., 4.];
    let tensor = Tensor::from_slice(&buf[..]);

    let tensor = tensor.reshape(&[2, 2]);
    let res = &tensor + &tensor;
    let res = &res * &tensor;

    let output = res.to_vec::<f32>();
    assert_eq!(output, vec![2.0, 8.0, 18.0, 32.0]);
    println!("Test passed! Result: {:?}", output);
}

#[test]
fn test_shape_preservation() {
    let buf = &[1., 2., 3., 4., 5., 6.];
    let tensor = Tensor::from_slice(&buf[..]);

    let tensor_2x3 = tensor.reshape(&[2, 3]);
    let res1 = &tensor_2x3 + &tensor_2x3;

    let realized1 = res1.realize();

    let res2 = &realized1 + &tensor_2x3;

    let output = res2.to_vec::<f32>();
    assert_eq!(output, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    println!("Test shape preservation passed! Result: {:?}", output);
}

#[test]
fn test_sum_reduce() {
    let buf = &[1., 2., 3., 4., 5., 6.];
    let tensor = Tensor::from_slice(&buf[..]);

    // Test sum of all elements
    let sum_all = tensor.sum(None);
    let output = sum_all.to_vec::<f32>();
    assert_eq!(output, vec![21.0]);
    println!("Sum all test passed! Result: {:?}", output);

    // Test sum along axis
    let tensor_2x3 = tensor.reshape(&[2, 3]);
    let sum_axis0 = tensor_2x3.sum(Some(0));
    let output = sum_axis0.to_vec::<f32>();
    assert_eq!(output, vec![5.0, 7.0, 9.0]);
    println!("Sum axis 0 test passed! Result: {:?}", output);

    let sum_axis1 = tensor_2x3.sum(Some(1));
    let output = sum_axis1.to_vec::<f32>();
    assert_eq!(output, vec![6.0, 15.0]);
    println!("Sum axis 1 test passed! Result: {:?}", output);
}

#[test]
fn test_max_reduce() {
    let buf = &[3., 1., 4., 1., 5., 9.];
    let tensor = Tensor::from_slice(&buf[..]);

    // Test max of all elements
    let max_all = tensor.max(None);
    let output = max_all.to_vec::<f32>();
    assert_eq!(output, vec![9.0]);
    println!("Max all test passed! Result: {:?}", output);

    // Test max along axis
    let tensor_2x3 = tensor.reshape(&[2, 3]);
    let max_axis0 = tensor_2x3.max(Some(0));
    let output = max_axis0.to_vec::<f32>();
    assert_eq!(output, vec![3.0, 5.0, 9.0]);
    println!("Max axis 0 test passed! Result: {:?}", output);
}

#[test]
fn test_min_reduce() {
    let buf = &[3., 1., 4., 2., 5., 0.];
    let tensor = Tensor::from_slice(&buf[..]);

    // Test min of all elements
    let min_all = tensor.min(None);
    let output = min_all.to_vec::<f32>();
    assert_eq!(output, vec![0.0]);
    println!("Min all test passed! Result: {:?}", output);
}

#[test]
fn test_reduce_fusion() {
    // Test that element-wise operations are fused with reduce operations
    let buf = &[1., 2., 3., 4.];
    let tensor = Tensor::from_slice(&buf[..]);

    // This should generate a single kernel with (a + a) fused into the sum reduction
    let doubled = &tensor + &tensor;
    let doubled_sum = doubled.sum(None);
    let output = doubled_sum.to_vec::<f32>();
    assert_eq!(output, vec![20.0]); // (1+1) + (2+2) + (3+3) + (4+4) = 20
    println!("Reduce fusion test passed! Result: {:?}", output);
}

#[test]
fn test_complex_reduce_fusion() {
    // Test complex combinations of reduces and binary operations
    let buf_a = &[1., 2., 3., 4., 5., 6.];
    let buf_b = &[2., 3., 4., 5., 6., 7.];

    let a = Tensor::from_slice(&buf_a[..]);
    let b = Tensor::from_slice(&buf_b[..]);

    // Test 1: Sum of element-wise multiplication
    let mul = &a * &b;
    let mul_sum = mul.sum(None);
    let output = mul_sum.to_vec::<f32>();
    assert_eq!(output, vec![112.0]); // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 = 2+6+12+20+30+42 = 112
    println!("Mul-sum fusion test passed! Result: {:?}", output);

    // Test 2: Mean of addition
    let add = &a + &b;
    let add_mean = add.mean(None);
    let output = add_mean.to_vec::<f32>();
    assert_eq!(output, vec![8.0]); // mean([3,5,7,9,11,13]) = 48/6 = 8
    println!("Add-mean fusion test passed! Result: {:?}", output);

    // Test 3: Max of subtraction (with reshape)
    let a_2x3 = a.reshape(&[2, 3]);
    let b_2x3 = b.reshape(&[2, 3]);
    let sub = &b_2x3 - &a_2x3;
    let sub_max = sub.max(Some(1)); // Max along axis 1
    let output = sub_max.to_vec::<f32>();
    assert_eq!(output, vec![1.0, 1.0]); // [[2-1, 3-2, 4-3], [5-4, 6-5, 7-6]] = [[1,1,1], [1,1,1]] -> max = [1,1]
    println!(
        "Sub-max with reshape fusion test passed! Result: {:?}",
        output
    );

    // Test 4: Chained operations - (a + b) * 2 then sum
    let two = Tensor::from_slice(&[2.0; 6]);
    let added = &a + &b;
    let multiplied = &added * &two;
    let complex = multiplied.sum(None);
    let output = complex.to_vec::<f32>();
    assert_eq!(output, vec![96.0]); // 2*(3+5+7+9+11+13) = 2*48 = 96
    println!(
        "Complex chained operations test passed! Result: {:?}",
        output
    );
}

#[test]
fn test_multi_axis_reduce() {
    // Test reducing different axes with binary ops
    let buf = &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let tensor = Tensor::from_slice(&buf[..]);

    // Reshape to 3D: 2x3x2
    let tensor_3d = tensor.reshape(&[2, 3, 2]);

    // Test different axis reductions with operations
    let doubled = &tensor_3d + &tensor_3d;

    // Sum along axis 0
    let sum_axis0 = doubled.sum(Some(0));
    let output = sum_axis0.to_vec::<f32>();
    assert_eq!(output, vec![16.0, 20.0, 24.0, 28.0, 32.0, 36.0]);
    // Shape becomes [3, 2], values are 2*([1,2]+[7,8]), 2*([3,4]+[9,10]), 2*([5,6]+[11,12])
    println!("3D sum axis 0 test passed! Result: {:?}", output);

    // Max along axis 1
    let max_axis1 = doubled.max(Some(1));
    let output = max_axis1.to_vec::<f32>();
    assert_eq!(output, vec![10.0, 12.0, 22.0, 24.0]);
    // Shape becomes [2, 2], max of each row across middle dimension
    println!("3D max axis 1 test passed! Result: {:?}", output);

    // Mean along axis 2
    let mean_axis2 = doubled.mean(Some(2));
    let output = mean_axis2.to_vec::<f32>();
    assert_eq!(output, vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0]);
    // Shape becomes [2, 3], mean of pairs
    println!("3D mean axis 2 test passed! Result: {:?}", output);
}

#[test]
fn test_mean_reduce() {
    let buf = &[1., 2., 3., 4., 5., 6.];
    let tensor = Tensor::from_slice(&buf[..]);

    // Test mean of all elements
    let mean_all = tensor.mean(None);
    let output = mean_all.to_vec::<f32>();
    assert_eq!(output, vec![3.5]);
    println!("Mean all test passed! Result: {:?}", output);

    // Test mean along axis
    let tensor_2x3 = tensor.reshape(&[2, 3]);
    let mean_axis1 = tensor_2x3.mean(Some(1));
    let output = mean_axis1.to_vec::<f32>();
    assert_eq!(output, vec![2.0, 5.0]);
    println!("Mean axis 1 test passed! Result: {:?}", output);
}

#[test]
fn test_reduce_with_device() {
    // Test reduce operations with device transfers
    let buf = &[1., 2., 3., 4., 5., 6.];
    let tensor = Tensor::from_slice(&buf[..]);

    // Transfer to CUDA and perform reduce
    let cuda_tensor = tensor.to_device(DeviceType::CUDA);
    let sum = cuda_tensor.sum(None);

    // Target device should be CUDA
    let target_device = sum.uop.get_target_device();
    assert_eq!(target_device, DeviceType::CUDA);

    // Only try to realize if CUDA is available
    {
        let output = sum.to_vec::<f32>();
        assert_eq!(output, vec![21.0]);
        println!("Reduce with device test passed! Result: {:?}", output);
    }

    {
        // Without CUDA, we just verify the graph construction
        println!(
            "Reduce with device test passed (CUDA not available, graph construction verified)!"
        );
    }
}

#[test]
fn test_sum_2d_cpu() {
    // Test sum on 2D tensor like in the benchmark
    let size = 128;
    let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&data);
    let tensor_2d = tensor.reshape(&[size, size]);

    let sum = tensor_2d.sum(None);
    let output = sum.to_vec::<f32>();

    let expected: f32 = (0..size * size).map(|i| i as f32).sum();
    assert_eq!(output[0], expected);
    println!("CPU 2D sum test passed! Result: {}", output[0]);
}

#[test]
fn test_sum_2d_cuda() {
    // Test sum on 2D tensor on CUDA like in the benchmark
    let size = 128;
    let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&data);
    let tensor_2d = tensor.reshape(&[size, size]);
    let tensor_cuda = tensor_2d.to_device(DeviceType::CUDA);

    let sum = tensor_cuda.sum(None);
    let output = sum.to_vec::<f32>();

    let expected: f32 = (0..size * size).map(|i| i as f32).sum();
    assert_eq!(output[0], expected);
    println!("CUDA 2D sum test passed! Result: {}", output[0]);
}

#[test]
fn test_spawn_realize_single() {
    let buf = &[1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_slice(buf);
    let tensor_cuda = tensor.to_device(DeviceType::CUDA);

    let result = (&tensor_cuda + &tensor_cuda);

    let handle = result.spawn_realize().expect("Failed to spawn realize");

    let realized = futures::executor::block_on(handle).expect("Failed to realize");

    let output: Vec<f32> = realized.to_vec();
    assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
    println!("Async single operation test passed! Result: {:?}", output);
}

#[test]
fn test_spawn_realize_multiple_join() {
    let buf1 = &[1.0f32, 2.0, 3.0, 4.0];
    let buf2 = &[5.0f32, 6.0, 7.0, 8.0];
    let buf3 = &[10.0f32, 20.0, 30.0, 40.0];

    let t1 = Tensor::from_slice(buf1).to_device(DeviceType::CUDA);
    let t2 = Tensor::from_slice(buf2).to_device(DeviceType::CUDA);
    let t3 = Tensor::from_slice(buf3).to_device(DeviceType::CUDA);

    let op1 = &t1 + &t1; // [2, 4, 6, 8]
    let op2 = &t2 * &t2; // [25, 36, 49, 64]
    let op3 = &t3 - &t1; // [9, 18, 27, 36]

    let handle1 = op1.spawn_realize().expect("Failed to spawn op1");
    let handle2 = op2.spawn_realize().expect("Failed to spawn op2");
    let handle3 = op3.spawn_realize().expect("Failed to spawn op3");

    let (result1, result2, result3) =
        futures::executor::block_on(async { futures::join!(handle1, handle2, handle3) });

    let output1: Vec<f32> = result1.expect("Op1 failed").to_vec();
    let output2: Vec<f32> = result2.expect("Op2 failed").to_vec();
    let output3: Vec<f32> = result3.expect("Op3 failed").to_vec();

    assert_eq!(output1, vec![2.0, 4.0, 6.0, 8.0]);
    assert_eq!(output2, vec![25.0, 36.0, 49.0, 64.0]);
    assert_eq!(output3, vec![9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn test_spawn_realize_complex_graph() {
    // Test async with complex computation graph
    let buf = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_slice(buf).to_device(DeviceType::CUDA);

    // Create complex operations
    let doubled = &tensor + &tensor;
    let squared = &tensor * &tensor;
    let combined = &doubled * &squared; // (a+a) * (a*a) = 2a * a^2 = 2a^3

    // Spawn async
    let handle = combined.spawn_realize().expect("Failed to spawn");

    // Await
    let result = futures::executor::block_on(handle).expect("Failed to realize");
    let output: Vec<f32> = result.to_vec();

    // Expected: 2*1^3=2, 2*2^3=16, 2*3^3=54, 2*4^3=128, 2*5^3=250, 2*6^3=432
    assert_eq!(output, vec![2.0, 16.0, 54.0, 128.0, 250.0, 432.0]);
}

#[test]
fn test_spawn_realize_with_reduce() {
    let buf = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_slice(buf).to_device(DeviceType::CUDA);

    let tensor_2x3 = tensor.reshape(&[2, 3]);
    let sum = tensor_2x3.sum(None);

    let handle = sum.spawn_realize().expect("Failed to spawn");

    let result = futures::executor::block_on(handle).expect("Failed to realize");
    let output: Vec<f32> = result.to_vec();

    assert_eq!(output, vec![21.0]);
}

#[test]
fn test_spawn_realize_many_concurrent() {
    let mut handles = Vec::new();

    for i in 0..10 {
        let val = (i as f32 + 1.0) * 10.0;
        let buf = vec![val; 100];
        let tensor = Tensor::from_slice(&buf).to_device(DeviceType::CUDA);
        // to_device() now realizes and copies to GPU, so buf can be dropped

        let result = &tensor + &tensor; // Should double each value
        let handle = result.spawn_realize().expect("Failed to spawn");
        handles.push((handle, val * 2.0));
    }

    let results = futures::executor::block_on(async {
        futures::future::try_join_all(handles.into_iter().map(|(h, _)| h)).await
    })
    .expect("Failed to join all");

    for (i, result) in results.iter().enumerate() {
        let output: Vec<f32> = result.to_vec();
        let expected = (i as f32 + 1.0) * 20.0;
        assert_eq!(output[0], expected);
        assert_eq!(output.len(), 100);
    }
}

#[test]
fn test_to_device_realizes_parent(){
    use axler_uop::UOp;

    // Test that to_device() realizes unrealized computations first
    let tensor = Tensor::from_slice(&[1., 2., 3., 4.]);
    let tensor1 = Tensor::from_slice(&[1., 2., 3., 4.]);

    let cpu_res = &tensor1 + &tensor; // Unrealized CPU computation

    // to_device() should realize cpu_res on CPU first, then copy to GPU
    let cpu_to_gpu = cpu_res.to_device(DeviceType::CUDA);

    // Verify it's now a Load wrapping a realized Kernel
    match &cpu_to_gpu.uop {
        UOp::Load(parent, device) => {
            assert_eq!(*device, DeviceType::CUDA);
            assert!(matches!(parent.as_ref(), UOp::Kernel(_, _, _, DeviceType::CPU)));
        }
        _ => panic!("Expected Load node"),
    }

    let gpu_tensor = Tensor::from_slice(&[1., 2., 3., 4.]).to_device(DeviceType::CUDA);
    let res = (&gpu_tensor + &cpu_to_gpu).realize();

    let res: Vec<f32> = res.to_vec();
    assert_eq!(res, vec![3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_kernel(){
    let t: Vec<f32> = Tensor::from_slice(&[1., 2., 3., 4.]).reshape(&[2, 2]).to_device(DeviceType::CUDA).sum(Some(1)).realize().to_vec();

    // Should preserve the reshape??
    println!("Res: {t:?}");
    // let t = Tensor::from_slice(&[1., 2., 3., 4.]).to_device(DeviceType::CUDA);
}
