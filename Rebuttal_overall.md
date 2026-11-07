# Rebuttal Draft (overall)

> 1. Insufficient ablations to isolate the benefits of variable-length grouping.

We conduct experiments on training-based spatially local 2D pooling and here are the results on LLaVA 1.5:

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
| VLVLM (ours) | **55.05** | **15.6** | 20.1 | **17.59** | **10.94** | **84.59%** |

The results demonstrate that 2D fixed pooling performs similarly as 1D fixed pooling, and our VLVLM maintains the best OCR-heavy task performance.

> 2. limited coverage of competitive baselines

In addition to *prompt-agnostic* PruMerge and PruneSID, we also conduct experiments with one *prompt-aware* training-free baseline, SparseVLM. Here are the results on LLaVA 1.5:

| method | TextVQA | OCRBench | OCRBenchv2 | DocVQA | ChartQAPro | Avg. Ret. (OCR) |
| ------ | ------- | -------- | ---------- | ------ | ---------- | --------------- |
| LLaVA-1.5-7B | 58.16 | 18.5 | 23.8 | 25.61 | 12.04 | 100% |
| 4x compression | 
| **SparseVLM** | 55.10 | 15.2 | **21.3** | 17.33 | **11.46** | 85.85% |
| VLVLM (ours) | **55.84** | **16.5** | 20.9 | **20.39** | 10.91 | **88.65%** |
| 8x compression |
| **SparseVLM** | 52.93 | 11.0 | 18.2 | 11.62 | **10.98** | 72.70% |
| VLVLM (ours) | **55.05** | **15.6** | **20.1** | **17.59** | 10.94 | **84.59%** |
 
Note that our VLVLM outperforms SparseVLM method, where we observe significant gains under aggressive compression budget (8x).


> 3. benchmark-level latency measures on Qwen3VL

We report LLM prefill latency in miliseconds for Qwen3VL on multiple benchmarks as follows. Our method significantly reduces LLM prefill latency and achieves similar speedup as other state-of-the-art methods like PruneSID.

| method | OCRBench | TextVQA | ChartQAPro |
| ------ | -------- | ------- | ---------- |
| Qwen3VL | 92.02 | 69.50 | 131.79 |
| 4x compression | 
| PruneSID | 57.77 | 53.54 | 63.62 |
| **VLVLM (ours)** | 58.45 | 54.00 | 63.02 |


> 4. Weak benchmark performance.

In addition to five benchmarks in the main text, we also evaluate our method VLVLM against baselines on a more recent, real-world, high-resolution benchmark - MME-RealWorld:

| Model | monitoring | OCR | Diagram and Table | Autonomous Driving | Remote Sensing | Overall | Avg. Ret. |
| ----- | ---------- | --- | ----------------- | ------------------ | -------------- | ------- | --------- |
| LLaVA | 19.22 | 43.62 | 20.21 | 26.89 | 22.74 | 28.09 | 100% |
| 4x compression |
| PruMerge | 19.31 | 42.65 | 20.27 | 26.94 | 22.63 | 27.87 | 99.22% |
| PruneSID | 19.03 | 43.14 | 20.61 | 26.64 | **28.09** | 28.12 | 100.11% |
| fixed pooling | **21.95** | 44.16 | 20.43 | 26.20 | 24.48 | 28.69 | 102.14% |
| VLVLM (ours) | 20.26 | **48.24** | **21.79** | **27.49** | 26.86 | **30.77** | **109.54%** |
| 8x compression |
| PruMerge | 19.31 | 42.46 | 20.27 | 26.97 | 22.66 | 27.82 | 99.04% |
| PruneSID | 19.31 | 42.89 | 20.49 | 26.64 | 22.74 | 28.03 | 99.79% |
| fixed pooling | 20.95 | **47.06** | 20.89 | 27.35 | 26.57 | 30.20 | 107.51% | 
| VLVLM (ours) | **21.86** | 46.95 | **21.44** | **27.43** | **26.97** | **30.44** | **108.37%** |

VLVLM outperforms other baselines such as PruMerge, PruneSID, and fixed pooling, under both moderate (i.e., 4$\times$) and aggressive (i.e., 8$\times$) compression budget, especially on tasks subsets like "OCR" and "Diagram and Table".


> 5. robustness of the incremental gains.

For all experiments, we already apply McNemar's test to assess whether the observed performance improvements over the baselines are statistically significant to measure the robustness of the incremental gains over trained fixed pooling. We reveal the *raw test statistics* and corresponding *p-values* for LLaVA 1.5 experiment as a more detailed example:

| method | TextVQA | OCRBench | OCRBenchv2 | DocVQA | ChartQAPro |
| ------ | ------- | -------- | ---------- | ------ | ---------- |
| LLaVA-1.5-7B | 58.16 | 18.5 | 23.8 | 25.61 | 12.04 | 100% |
| 4x compression | 
| fixed pooling | 55.74 | 15.7 | 20.9 | 18.78 | **11.10** |
| VLVLM (ours) | **55.84** | **16.5** | 20.9 | **20.39** | 10.91 |
| McNemar's test | 0.0733 | 2.127 | 0.237 | 30.30 | 0.179 |
| p-value | 0.786 | 0.145 | 0.626 | ***0.000*** | 0.672 |
| 8x compression |
| fixed pooling | 54.05 | 14.3 | 19.5 | 16.00 | 10.81 |
| VLVLM (ours) | **55.05** | **15.6** | 20.1 | **17.59** | **10.94** |
| McNemar's test | 1.481 | 3.25 | 0.575 | 32.397 | 0.455 |
| p-value | 0.223 | 0.071 | 0.448 | ***0.000*** | 0.499 |

On benchmark DocVQA in particular, our method VLVLM's performance gain is consistently statistically significant.


> 6. incomplete comparisons with training-based methods.

Most mentioned training-based baselines have their own limitations. *TokenPacker* is an alternative to the MLP projector in VLM architectures such as LLaVA and QwenVL. It is meant to replace the original projector in the VLM, which does not share the same purpose of our work. *LearnPruner*, *MetaCompress*, and *EvoComp* do not have open-sourced code implementation or artifacts, meaning that their results cannot be reproduced.


