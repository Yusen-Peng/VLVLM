## General Comments

### A broader coverage of training-free and training-based baselines

For training-free baselines, in addition to *prompt-agnostic* PruMerge and PruneSID, we also conduct experiments with one *prompt-aware* training-free baseline, SparseVLM. Here are the results on LLaVA 1.5:

| method | TextVQA | OCRBench | OCRBenchv2 | DocVQA | ChartQAPro | Avg. Ret. (OCR) |
| ------ | ------- | -------- | ---------- | ------ | ---------- | --------------- |
| LLaVA-1.5-7B | 58.16 | 18.5 | 23.8 | 25.61 | 12.04 | 100% |
| 4x compression | 
| SparseVLM | 55.10 | 15.2 | 21.3 | 17.33 | 11.46 | 85.85% |
| VLVLM (ours) | 55.84 | 16.5 | 20.9 | 20.39 | 10.91 | 88.65% |
| 8x compression |
| SparseVLM | 52.93 | 11.0 | 18.2 | 11.62 | 10.98 | 72.70% |
| VLVLM (ours) | 55.05 | 15.6 | 20.1 | 17.59 | 10.94 | 84.59% |
 
Note that our VLVLM consistently outperforms SparseVLM method, where we observe significant gains under aggressive compression budget (8x). 

For training-based baselines, other than our default 1D fixed pooling, we extend our experiments on spatially local 2D pooling. Here are the results on LLaVA 1.5:

| method | TextVQA | OCRBench | OCRBenchv2 | DocVQA | ChartQAPro | Avg. Ret. (OCR) |
| ------ | ------- | -------- | ---------- | ------ | ---------- | --------------- |
| LLaVA-1.5-7B | 58.16 | 18.5 | 23.8 | 25.61 | 12.04 | 100% |
| 4x compression | 
| fixed pooling | 55.74 | 15.7 | 20.9 | 18.78 | 11.10 | 86.80% |
| fixed (2D) | 53.37 | 15.7 | 21.9 | 18.60 | 10.67 | 85.97% |
| VLVLM (ours) | 55.84 | 16.5 | 20.9 | 20.39 | 10.91 | 88.65% |
| 8x compression |
| fixed pooling | 54.05 | 14.3 | 19.5 | 16.00 | 10.81 | 80.88% |
| fixed (2D) | 51.54 | 13.6 | 20.4 | 16.30 | 9.39 | 77.89% |               
| VLVLM (ours) | 55.05 | 15.6 | 20.1 | 17.59 | 10.94 | 84.59% |

The results demonstrate that 2D fixed pooling performs similarly as 1D fixed pooling, and our VLVLM maintains the best OCR-heavy task performance.

### Prefill latency improvement

We measure the LLM prefill latency to demonstrate the efficiency gain beyond FLOPs, spanning a diverse collection of benchmarks (TextVQA, DocVQA for OCR-heavy tasks, MME, GQA, POPE, AND MM-Vet or general VQA tasks): 

| Method | MME | TextVQA | DocVQA | GQA | POPE | MM-Vet |
| ------ | --- | ------- | ------ | --- | ---- | ------ |
| LLaVA-1.5 7B | 60.02 | 65.79 | 57.37 | 59.31 | 55.59 | 55.72 |
| 4x compression |
| PruMerge | 29.82 | 32.12 | 29.17 | 29.37 | 29.42 | 29.83 |
| PruneSID | 28.96 | 32.37 | 29.07 | 29.19 | 29.20 | 28.99 |
| SparseVLM | 42.44 | 46.34 | 41.03 | 41.65 | 42.75 | 41.81 |
| Fixed pooling | 29.81 | 33.34 | 29.93 | 30.03 | 30.03 | 29.73 |
| VLVLM | 29.62 | 33.34 | 30.00 | 30.23 | 29.78 | 29.82 |

Under 4x compression, most methods and baselines including VLVLM can reach up to **2x** prefill latency speedup with a single batch size; SparseVLM only reaches up to **1.5x** latency boost since it starts merging image tokens during the LLM layer.


To validate the latency gains with more modern VLMs, we further measure the prefill latency improvement on multiple benchmarks including OCRBench, TextVQA, and ChartQAPro for Qwen3VL 4B model. We first conduct a profiling experiment on a single-sample (batch size = 1) setting, and VLVLM achieves similar prefill speedup as other baselines such as fixed pooling and PruneSID. To maximize GPU utilization, we also consider batched (batch size = 4) setting for inference, which is not supported by PruneSID. 
VLVLM accelerates prefill latency similar to fixed pooling.

| Method | OCRBench | TextVQA | ChartQAPro | OCRBench | TextVQA | ChartQAPro |
| ------ | -------- | ------- | ---------- | -------- | ------- | ---------- |
| batch size = 1 | | | | batch size = 4 |
| Qwen3VL-4B | 92.02 | 69.50 | 131.79 | 406.78 | 299.09 | 830.45 |
| 4x compression |
| PruneSID | 57.77 | 53.54 | 63.62 | -- | -- | -- |
| Fixed pooling | 57.76 | 53.87 | 63.49 | 123.85 | 147.11 | 216.54 |
| VLVLM | 58.45 | 54.00 | 63.02 | 124.36 | 143.52 | 215.41 |


### Evaluation on high-resolution benchmarks

We also evaluate our method VLVLM against baselines on a more recent, real-world, high-resolution benchmark MME-RealWorld [1].

| Model | monitoring | OCR | Diagram and Table | Autonomous Driving | Remote Sensing | Overall | Avg. Ret. |
| ----- | ---------- | --- | ----------------- | ------------------ | -------------- | ------- | --------- |
| LLaVA | 19.22 | 43.62 | 20.21 | 26.89 | 22.74 | 28.09 | 100% |
| 4x compression |
| PruMerge | 19.31 | 42.65 | 20.27 | 26.94 | 22.63 | 27.87 | 99.22% |
| PruneSID | 19.03 | 43.14 | 20.61 | 26.64 | 28.09 | 28.12 | 100.11% |
| fixed pooling | 21.95 | 44.16 | 20.43 | 26.20 | 24.48 | 28.69 | 102.14% |
| VLVLM (ours) | 20.26 | 48.24 | 21.79 | 27.49 | 26.86 | 30.77 | 109.54% |
| 8x compression |
| PruMerge | 19.31 | 42.46 | 20.27 | 26.97 | 22.66 | 27.82 | 99.04% |
| PruneSID | 19.31 | 42.89 | 20.49 | 26.64 | 22.74 | 28.03 | 99.79% |
| fixed pooling | 20.95 | 47.06 | 20.89 | 27.35 | 26.57 | 30.20 | 107.51% | 
| VLVLM (ours) | 21.86 | 46.95 | 21.44 | 27.43 | 26.97 | 30.44 | 108.37% |

VLVLM outperforms other baselines such as PruMerge, PruneSID, and fixed pooling, under both moderate (4x) and aggressive (8x) compression budgets, especially on tasks subsets like "OCR" and "Diagram and Table".

### Coarse-grained benchmark results for Qwen3VL 

We evaluate our Qwen3VL experiment on coarse-grained general VQA tasks, covering benchmarks such as MMBench, MME, GQA, and POPE. We observe that on these simple benchmarks, all methods including VLVLM and representative baselines are performing well, revealing applicability on modern state-of-the-art VLMs.

| Method | MMBench | MME | GQA | POPE |
| ------ | ------- | --- | --- | ---- |
| Qwen3VL-4B | 83.90 | 1710.48 | 62.49 | 89.66 |
| SFT@66K | 88.42 | 1729.54 | 62.94 | 90.22 |
| Additional 4x compression |
| PruneSID | 85.63 | 1638.41 | 61.13 | 89.35 |
| Fixed pooling@66K | 85.31 | 1668.44 | 60.41 | 88.90 |
| VLVLM@66K | 84.97 | 1680.89 | 59.54 | 89.22 |

### References

[1] MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?, ICLR 2025.

