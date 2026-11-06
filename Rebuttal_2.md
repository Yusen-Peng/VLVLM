We thanks for reviewer ``FQoK`` feedback and comments. Here is how we address the raised weaknesses and concerns:

> Incomplete comparisons with training-based methods. VisionSelector[2], MetaCompress[3], and EvoComp[4], are omitted, making it difficult to assess the effectiveness of VLVLM against existing alternatives. Additionally, why was the fine-tuned variant of PruMerge not considered?

[TO-THINK]

TokenPacker - an alternative projector that replaces the MLP projector, not generalizable approach? also flash attention??



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


> Weak benchmark performance and severe degradation on modern VLMs such as Qwen3VL

[TO-DO] - add more general VQA benchmarks for Qwen3VL to validate?

also clarify OCR is hard?


> benchmark-level latency measures.

[TO-DO]

