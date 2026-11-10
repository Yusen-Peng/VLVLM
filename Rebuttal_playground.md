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
