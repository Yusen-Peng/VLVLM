We thanks for reviewer ``FQoK`` feedback and comments. Here is how we address the raised weaknesses and concerns:

> Incomplete comparisons with training-based methods.

Most mentioned training-based baselines have their own limitations. *TokenPacker* is an alternative to the MLP projector in VLM architectures such as LLaVA and QwenVL. It is meant to replace the original projector in the VLM, which does not share the same purpose of our work. *LearnPruner*, *MetaCompress*, and *EvoComp* do not have open-sourced code implementation or artifacts, meaning that their results cannot be reproduced.



> Insufficient ablations to isolate the benefits of variable-length grouping.

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


> Weak benchmark performance.

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

> severe degradation on modern VLMs such as Qwen3VL.

In terms of degradation on modern VLMs such as Qwen3VL, since we target quite challenging tasks that involve OCR-heavy perception, we do expect performance degradation on these benchmarks with an additional moderate 4x compression. However, our method VLVLM is still able to outperform multiple state-of-the-art baselines, where we also add experiments of CDPruner in addition to PruneSID in the initial submission:

| Model | TextVQA | OCRBench | OCRBenchv2 | ChartQAPro | Avg. Ret. | 
| ----- | ------- | -------- | ---------- | ---------- | --------- |
| Qwen3VL-4B | 77.99 | 85.5 | 68.4 | 26.57 | -- |
| +SFT@ 65K | 81.49 | 83.9 | 67.2 | 27.89 | 100% |
| 4x compression |
| PruneSID | 71.23 | 49.9 | 46.7 | **23.16** | 74.85% |        
| **CDPruner** | 73.15 | 51.8 | 49.1 | 18.98 | 73.16% | 
| fixed pooling@65K | 70.50 | 51.1 | **51.3** | **19.58** | 73.49% |
| VLVLM@65K | **76.20** | **54.6** | 51.2 | 19.08 | **75.80%** |

> benchmark-level latency measures.

[COPY-PASTE]

