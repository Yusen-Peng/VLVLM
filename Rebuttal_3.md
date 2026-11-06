We thanks for reviewer ``u9A8`` feedback and comments. Here is how we address the raised weaknesses and concerns:

> The incremental methodological contribution is insufficiently established: the main changes appear to be placement after the vision encoder and budget-controlled top-K inference. 

[TO-WRITE]

> benchmark-level latency measures (Please specify hardware, batch size, input resolution, and output length, and clarify the incremental training cost relative to fixed pooling.)

[TO-DO]


> comparison with more training-based methods.


> Robustness of the incremental gains.

For all experiments, we already apply McNemar's test to assess whether the observed performance improvements over the baselines are statistically significant to measure the robustness of the incremental gains over trained fixed pooling. We reveal the raw test statistics and corresponding p-values for LLaVA 1.5 experiment as a more detailed example:

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

On benchmark DocVQA in particular, our method VLVLM's performance gain is consistently significant.

Besides statistical testing, we also evaluate our method against baselines on a more recent, real-world, high-resolution benchmark: MME-RealWorld:

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
| VLVLM (ours) | **21.86** | 46.95 | **21.44** | **27.43** | **26.97** | **30.44** | 108.37% |



> limited coverage of competitive baselines

In addition to *prompt-agnostic* PruMerge and PruneSID, we also conduct experiments with one *prompt-aware* training-free baseline, SparseVLM. Here are the results on LLaVA 1.5:

| method | TextVQA | OCRBench | OCRBenchv2 | DocVQA | ChartQAPro | Avg. Ret. (OCR) |
| ------ | ------- | -------- | ---------- | ------ | ---------- | --------------- |
| LLaVA-1.5-7B | 58.16 | 18.5 | 23.8 | 25.61 | 12.04 | 100% |
| 4x compression | 
| SparseVLM | 55.10 | 15.2 | **21.3** | 17.33 | **11.46** | 85.85% |
| VLVLM (ours) | **55.84** | **16.5** | 20.9 | **20.39** | 10.91 | **88.65%** |
| 8x compression |
| SparseVLM | 52.93 | 11.0 | 18.2 | 11.62 | **10.98** | 72.70% |
| VLVLM (ours) | **55.05** | **15.6** | **20.1** | **17.59** | 10.94 | **84.59%** |
 
Note that our VLVLM outperforms SparseVLM method, where we observe significant gains under aggressive compression budget (8x).

