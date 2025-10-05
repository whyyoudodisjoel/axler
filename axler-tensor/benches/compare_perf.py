from tinygrad import Tensor
import time
import numpy as np
t = Tensor.rand(50, 50).to("CPU")
# Warmup
for _ in range(5):
    _ = (t + t).numpy()
# Measure
times = []
for _ in range(10):
    start = time.perf_counter_ns()
    result = (t + t).realize()
    end = time.perf_counter_ns()
    times.append(end - start)
    
print(int(np.median(times)))
print(f"Sh Tinygrad: {t.shape}")

import torch
t = torch.rand(50, 50).to("cpu")
# Warmup
for _ in range(5):
    _ = (t + t).cpu().numpy()
# Measure
times = []
for _ in range(10):
    start = time.perf_counter_ns()
    result = t + t
    end = time.perf_counter_ns()
    times.append(end - start)

print(int(np.median(times)))
print(f"Sh Torch: {t.shape}")
