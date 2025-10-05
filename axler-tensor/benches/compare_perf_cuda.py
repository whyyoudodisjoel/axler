from tinygrad import Tensor
import time
import numpy as np

times = []
for _ in range(10):
    start = time.perf_counter_ns()
    t = Tensor.rand(128).to("CUDA")
    result = (t + t).realize()
    end = time.perf_counter_ns()
    times.append(end - start)
    
print(int(np.median(times)))
print(f"Sh Tinygrad: {t.shape}")

import torch

# Measure
times = []
for _ in range(10):
    start = time.perf_counter_ns()
    t = torch.rand(128).to("cuda")

    result = t + t
    end = time.perf_counter_ns()
    times.append(end - start)

print(int(np.median(times)))
print(f"Sh Torch: {t.shape}")
