## accumulate_confidence

```python
import torch
from cuda_utils._C import accumulate_gaussian_confidence

H = 600
W = 800
P = 1000

indexes = torch.arange(-1,P)
index_map = torch.randint(0,P, (H,W), dtype=torch.int32).cuda()
confidence_map = torch.rand(H,W).cuda() ** 2
confidence_max, confidence_min, confidence_mean = accumulate_gaussian_confidence(H, W, P, index_map, confidence_map)
```