We thanks for reviewer ``yzVq`` feedback and comments. Here is how we address the raised weaknesses and concerns:

> implementation about the terminal boundary and the straight-through estimator.

For the terminal boundary, in order to match the exact budget, we allow the last segment to be dropped if the very last image token is not being selected as a boundary. Besides, in our code implementation, the straight-through estimator is only used during training, where we do:

```python
hard_boundaries = (soft_boundaries > 0.5).float()
hard_boundaries = hard_boundaries - soft_boundaries.detach() + soft_boundaries
```

Note that after thresholding during training, we eventually obtain the hard boundaries via the standard straight-through estimator.

> interpretation about text coverage analysis and experiments on spatially local 2D pooling.

Although our proposed VLVLM leads to background mixing due to the nature of token pooling, many more boundary tokens still end up being predicted around its regions, preserving its strong OCR-related task performance. Additionally, mixing the background is not necessarily a weakness: depending on the question prompt, it will be beneficial not to discard the background completely.

We also conduct experiments on training-based spatially local 2D pooling and here are the results on LLaVA 1.5:

| method | TextVQA | OCRBench | OCRBenchv2 | DocVQA | ChartQAPro | Avg. Ret. (OCR) |
| ------ | ------- | -------- | ---------- | ------ | ---------- | --------------- |
| LLaVA-1.5-7B | 58.16 | 18.5 | 23.8 | 25.61 | 12.04 | 100% |
| 4x compression | 
| fixed pooling | 55.74 | 15.7 | 20.9 | 18.78 | **11.10** | 86.80% |
| **fixed (2D)** | 53.37 | 15.7 | **21.9** | 18.60 | 10.67 | 85.97% |
| VLVLM (ours) | **55.84** | **16.5** | 20.9 | **20.39** | 10.91 | **88.65%** |
| 8x compression |
| fixed pooling | 54.05 | 14.3 | 19.5 | 16.00 | 10.81 | 80.88% | 
| **fixed (2D)** | 51.54 | 13.6 | **20.4** | 16.30 | 9.39 | 77.89% |               
| VLVLM (ours) | **55.05** | **15.6** | 20.1| **17.59** | **10.94** | **84.59%** |

The results demonstrate that 2D fixed pooling performs similarly as 1D fixed pooling, and our VLVLM maintains the best OCR-heavy task performance.

> comparison with more training-based methods such as finetuned PruMerge and PruneSID or LearnPruner.

[TO-THINK]

> benchmark-level latency measures.


Because Qwen3VL processes images at their native resolution without resizing, the efficiency improvements are more pronounced: for an image with more than 3700 patches, TFLOPs drop from 40.3 to 17.4 with $4\times$ compression.







Qwen3VL benchmark runtime (I/O bound):

| method | OCRBench | ChartQAPro |
| ------ | -------- | ---------- |
| original | 483 sec / 1000 samples | 893 sec / 1948 samples |
| 4x compression |
| PruneSID | 460 sec / 1000 samples | 872 sec / 1948 samples | 
| VLVLM | 470 sec / 1000 samples | 844 sec / 1948 samples |

