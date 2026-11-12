# Rebuttal: Learning Variable Length Visual Tokens for Efficient Vision Language Models

## General Comments

### Novelty and contribution

We want to emphasize the fact we observe that adopting the similar dynamic token pooling idea from NLP (e.g., byte-level LMs) to vision is *not trivial*: in fact, **all of our design decisions are made specific to visual compression**. We elaborate on them as follows.


1. Putting the boundary predictor after the vision encoder instead of within the vision encoder: (1) most of the vision encoders are already pretrained and kept frozen during multimodal instruction tuning. Importantly, our early experiments show that inserting a boundary predictor within the vision encoder severely *degrades the performance* (e.g., almost 30% drop on MMBench); (2) in modern state-of-the-art VLM architectures such as Qwen3VL, they use **DeepStack [1]** to stack in image features from shallow layers: a boundary predictor in between layers would lead to *shape mismatch*; (3) The main compute bottleneck in terms of both FLOPs and latency actually stem from *LLM layers*; saving computation within the vision encoder would only lead to very limited additional efficiency gains.


2. Using top-K selection during inference: (1) the original dynamic token pooling work in NLP does not use top-K during inference, where instead it uses the same thresholding approach during its training. That is because the predicted boundary probabilities are quite confident (either close 1s or 0s) and applying thresholding leads to similar compute budget. However, in our early experiments with LLaVA, we find that the predicted boundary probabilities are typically *not confident enough*; therefore, only top-K selection can guarantee the matched budget when comparing with baselines. We speculate that this may be lie in the attention pattern difference between *causal-attention LMs* and *bidirectional-attention vision encoders*. (2) using top-K selection can also allow for *any arbitrary inference budget*, as we have demonstrated in our paper's discussion section "you only train it once". This is very valuable since one can just train one VLVLM once, and set any inference budget on the fly. This particular benefit of achieving arbitrary inference budget is never enabled in the dynamic token pooling work and it's only possible when we use top-K selection during inference.

We also want to highlight our contribution *beyond algorithm and methodology*. (1) performance-wise, our proposed method is effective to improve VLM's capabilities on text-centric and fine-grained benchmarks, spanning different vision encoders, underlying LLMs, and VLM architectures; (2) inference efficiency-wise, our VLVLM achieves similar FLOP and latency gains as state-of-the-art baselines and the boundary predictor only introduces, as Table 7 in the appendix shows, 4.2M learned parameters on top of fixed pooling; (3) data efficiency-wise, as our discussion section "Can we Train with Fewer Samples?" illustrates, VLVLM achieves the same level of performance retention while using only **25%** of the data required by fixed pooling (VLVLM @ 166.2K v.s. fixed 665K). Note that all of these contributions are also distinct benefits specific to visual compression.


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

[MAYBE-ADD-TOKENPACKER-HERE]

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

| Method | OCRBench | TextVQA | ChartQAPro | OCRBench | TextVQA | ChartQAPro |
| ------ | -------- | ------- | ---------- | -------- | ------- | ---------- |
| batch size = 1 | | | | batch size = 4 |
| Qwen3VL-4B | 92.02 | 69.50 | 131.79 | 406.78 | 299.09 | 830.45 |
| 4x compression |
| PruneSID | 57.77 | 53.54 | 63.62 | -- | -- | -- |
| Fixed pooling | 57.76 | 53.87 | 63.49 | 123.85 | 147.11 | 216.54 |
| VLVLM | 58.45 | 54.00 | 63.02 | 124.36 | 143.52 | 215.41 |


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


### Evaluation on high-resolution benchmarks

We also evaluate our method VLVLM against baselines on a more recent, real-world, high-resolution benchmark MME-RealWorld [2].

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

VLVLM outperforms other baselines such as PruMerge, PruneSID, and fixed pooling, under both moderate (4x) and aggressive (8x) compression budgets, especially on tasks subsets like "OCR" and "Diagram and Table". More importantly, both our method VLVLM and fixed pooling baseline actually **exceed the performance of original, uncompressed LLaVA**, highlighting favorable practical use cases.

### References

[1] DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs, NeurIPS 2024.
[2] MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?, ICLR 2025.



-------------------------------- the end of General Comments --------------------------------


## for reviewer yzVq

We thanks for reviewer ``yzVq`` feedback and comments. Here is how we address the raised weaknesses and concerns:

> My first concern is the treatment of the final chunk in Algorithms 2–3 of Appendix B. Algorithm 2 takes the K highest-probability boundaries without requiring the last token to be a boundary. Algorithm 3 allocates segments from the boundary count and retains only indices 0 through S−1. For a one-image batch with boundaries [0,1,0,0], the shifted cumulative indices are [0,0,1,1], but S=1, leaving the final two tokens outside every output segment. The implementation may force a terminal boundary or handle the remainder separately, but this is not stated. Please provide implementation-faithful pseudocode that guarantees both full coverage and the exact budget, and specify how soft and hard boundaries connect to the pooling loss through the straight-through estimator.

For the terminal boundary, in order to match the exact budget, we actually allow the last segment to be dropped if the very last image token is not being selected as a boundary and it empirically affects little performance. Here is the exact code we use to match the inference budget:

```python
# top-k routing: select exactly k boundaries per sample
_, topk_idx = torch.topk(soft_boundaries, k=k, dim=-1)
hard_boundaries = torch.zeros_like(soft_boundaries)
hard_boundaries.scatter_(dim=-1, index=topk_idx, value=1.0)
```

Besides, in our code implementation, we emphasize that the straight-through estimator is only used during training, where we obtain the hard boundaries via the standard straight-through estimator in order to train VLVLM with non-differentiable operations (i.e., boundary selection):

```python
hard_boundaries = (soft_boundaries > 0.5).float()
hard_boundaries = hard_boundaries - soft_boundaries.detach() + soft_boundaries
```

> In my view, Table 5’s text coverage indicates where chunk endpoints fall, not directly how much text information survives pooling. VLVLM averages non-boundary tokens too; an endpoint on text does not guarantee that the corresponding chunk avoids substantial background mixing. The paper itself finds that PruneSID sometimes has higher coverage but worse task performance. I would like chunk-length and text/background-mixing statistics for text-containing chunks, plus trained random variable-length boundaries or spatially local 2D pooling controls. Appendix E already acknowledges vertical-text failures; checking row boundaries and scan order would make the semantic-boundary interpretation more convincing.

Although our proposed VLVLM leads to background mixing due to the nature of token pooling, many more boundary tokens still end up being predicted around text-heavy regions, preserving its strong OCR-related task performance. Additionally, mixing the background is *not necessarily a weakness*: depending on the question prompt, it will be beneficial not to discard the background completely [TO-CITE].

For spatially local 2D pooling controls, we also conduct experiments on training-based spatially local 2D pooling. Please feel free to refer to our General Comments "A broader coverage of training-free and training-based baselines", where our results demonstrate that 2D fixed pooling performs similarly as 1D fixed pooling, and our VLVLM maintains the best OCR-heavy task performance.


> The fixed-pooling control is valuable, but PruMerge and PruneSID are still evaluated only as training-free methods. Since training–inference mismatch is central to the motivation, adapting those selectors with the same data and update budget is an especially relevant control. Perceiver does not fully resolve this issue because it changes the compression mechanism as well. LearnPruner is discussed but not directly compared. I would like at least one nonuniform selection or merging baseline with matched fine-tuning, to establish whether the advantage comes from variable-length consecutive grouping rather than primarily from adapting the language model to compressed inputs.

[WAIT-ON-TOKENPACKER]


> The FLOPs, KV-cache, and memory results in Appendix C are useful, but Table 8’s 55-to-30-hour comparison mainly measures training with versus without FlashAttention, not inference acceleration over baselines using the same backend. Algorithm 3 also constructs a token-to-segment assignment matrix, and it is unclear whether the implementation uses a more efficient segmented reduction. Please report time to first token, end-to-end latency at fixed generation lengths, and the predictor/pooling overhead under matched settings, particularly for high-resolution Qwen3VL. This would establish whether the arithmetic savings translate into deployment benefits.

Please feel free to refer to our General Comments "Prefill latency improvement", where we measure the prefill latency gains on both LLaVA and Qwen3VL across multiple baselines and benchmarks. Under 4x compression, most methods and baselines including our VLVLM can reach up to **2x** prefill latency speedup, for both LLaVA 1.5 7B and Qwen3VL 4B.


## References






Thank you again for your effort to review our work. We hope you could spend some time reviewing our rebuttal, and we are always happy to further clarify anything needed. If our rebuttal can address your concerns, we would appreciate it if you may reflect that in an updated score.

-------------------------------- the end of responses to yzVq --------------------------------



## for reviewer FQoK

We thanks for reviewer ``FQoK`` feedback and comments. Here is how we address the raised weaknesses and concerns:

> Incomplete comparisons with training-based methods. The evaluation primarily compares against training-free methods, while the training-based baselines are limited to Perceiver and fixed pooling. Several relevant training-based visual-token compression methods, such as TokenPacker[1], VisionSelector[2], MetaCompress[3], and EvoComp[4], are omitted, making it difficult to assess the effectiveness of VLVLM against existing alternatives.

Most mentioned training-based baselines have their own limitations. *LearnPruner*, *MetaCompress*, and *EvoComp* do not have open-sourced code implementation or artifacts, meaning that their results cannot be reproduced. Additionally, [WAIT-ON-TOKENPACKER]


> Insufficient ablations to isolate the benefits of variable-length grouping. The fixed-pooling baseline operates on flattened visual tokens, potentially overlooking spatial locality. At the minimum, comparisons against trained 2D spatial pooling would help establish whether the improvements specifically arise from adaptive grouping.

For spatially local 2D pooling controls, we also conduct experiments on training-based spatially local 2D pooling. Please feel free to refer to our General Comments "A broader coverage of training-free and training-based baselines", where our results demonstrate that 2D fixed pooling performs similarly as 1D fixed pooling, and our VLVLM maintains the best OCR-heavy task performance.


> Weak benchmark performance and severe degradation on modern VLMs. The evaluation relies heavily on LLaVA-1.5-7B, which achieves only 15–30% on benchmarks such as OCRBench and DocVQA, substantially below the 80–90% performance achieved by modern VLMs. Improvements of just 1–2 points in such a low-performing regime raise questions about their practical significance. More concerningly, adapting Qwen3-VL-4B (Table 4) results in severe performance degradation, with OCRBench dropping from 83.9 to 54.6 despite fine-tuning. These results raise serious concerns about the applicability of the proposed method to modern VLMs.

In terms of "weak benchmark performance", we also evaluate our method VLVLM against baselines on a more recent, real-world, high-resolution benchmark called MME-RealWorld, and you can feel free to refer to our General Comments "Evaluation on high-resolution benchmarks". We observe that not only does VLVLM outperform other baselines especially on tasks subsets like "OCR" and "Diagram and Table", more importantly, our method VLVLM (and our proposed fixed pooling baseline) actually **exceed the performance of original, uncompressed LLaVA**, highlighting its favorable practical use cases.

As for "severe degradation on modern VLMs", we clarify that the perceived "severe" degradation stems from evaluating on difficult, find-grained, text-centric benchmarks only. In our General Comment "Coarse-grained benchmark results for Qwen3VL", we also evaluate our Qwen3VL experiment on coarse-grained general VQA tasks, covering benchmarks such as MMBench, MME, GQA, and POPE. We observe that on these simple benchmarks, all methods including VLVLM and representative baselines are performing well (almost lossless performance), revealing applicability on modern state-of-the-art VLMs.


> Qwen3-VL already employs a learned 2×2 spatial token merger and supports native-resolution inputs. Can the authors report actual time-to-first-token (TTFT) or end-to-end prefill latency on OCRBench (or any other benchmark), comparing the original Qwen3-VL with VLVLM under identical conditions? It would be useful to report the image resolutions, visual-token counts, and latency breakdown between vision encoding and LLM prefill to establish the practical benefits of additional compression beyond Qwen3-VL's existing token reduction.

Please feel free to refer to our General Comments "Prefill latency improvement", where we measure the prefill latency gains on both LLaVA and Qwen3VL across multiple baselines and benchmarks. Under 4x compression, most methods and baselines including our VLVLM can reach up to **2x** prefill latency speedup, for Qwen3VL 4B model.



Thank you again for your effort to review our work. We hope you could spend some time reviewing our rebuttal, and we are always happy to further clarify anything needed. If our rebuttal can address your concerns, we would appreciate it if you may reflect that in an updated score.


-------------------------------- the end of responses to FQoK --------------------------------


## for reviewer u9A8

We thanks for reviewer ``u9A8`` feedback and comments. Here is how we address the raised weaknesses and concerns:


> Contribution beyond existing dynamic pooling: Which design elements provide the substantive contribution beyond Nawrot et al.’s dynamic token pooling and related dynamic-chunking approaches? Beyond applying the mechanism after the vision encoder and using top-K inference, what existing evidence isolates a new capability or a distinct benefit specific to visual compression?

We want to emphasize the fact we observe that adopting the similar dynamic token pooling idea from NLP to vision is not trivial, where all of our design decisions are made specific to visual compression. Please feel free to refer to our General Comments "Novelty and contribution". We also highlight our contribution *beyond algorithm and methodology*, including benchmark performance, inference efficiency, and training data efficiency, elaborated in our General Comments.


> Robustness of the incremental gains: How do you explain the weaker or negative gains over trained fixed pooling at SigLIP2 4× and Qwen2.5-14B 10×, alongside the stronger gains at 8×? If available, please provide variation across training runs or other evidence supporting the robustness of this pattern.


For all experiments, we already apply McNemar's test to assess whether the observed performance improvements over the baselines are statistically significant to measure the robustness of the incremental gains over trained fixed pooling. We reveal the raw test statistics and corresponding p-values for LLaVA 1.5 experiment as a more detailed example:

| method | TextVQA | OCRBench | OCRBenchv2 | DocVQA | ChartQAPro |
| ------ | ------- | -------- | ---------- | ------ | ---------- |
| LLaVA-1.5-7B | 58.16 | 18.5 | 23.8 | 25.61 | 12.04 |
| 4x compression | 
| fixed pooling | 55.74 | 15.7 | 20.9 | 18.78 | 11.10 |
| VLVLM (ours) | **55.84** | **16.5** | 20.9 | **20.39** | 10.91 |
| McNemar's test | 0.0733 | 2.127 | 0.237 | 30.30 | 0.179 |
| p-value | 0.786 | 0.145 | 0.626 | ***0.000*** | 0.672 |
| 8x compression |
| fixed pooling | 54.05 | 14.3 | 19.5 | 16.00 | 10.81 |
| VLVLM (ours) | **55.05** | **15.6** | 20.1 | **17.59** | 10.94 |
| McNemar's test | 1.481 | 3.25 | 0.575 | 32.397 | 0.455 |
| p-value | 0.223 | 0.071 | 0.448 | ***0.000*** | 0.499 |

On benchmark DocVQA in particular, our method VLVLM's performance gain is consistently statistically significant.


> Measured inference efficiency: Can you report end-to-end inference latency, time to first token, and throughput against fixed pooling and the evaluated compression baselines, including boundary prediction, token grouping, and generation? Please specify hardware, batch size, input resolution, and output length, and clarify the incremental training cost relative to fixed pooling.

Please feel free to refer to our General Comments "Prefill latency improvement", where we measure the prefill latency gains on both LLaVA and Qwen3VL across multiple baselines and benchmarks. Under 4x compression, most methods and baselines including our VLVLM can reach up to **2x** prefill latency speedup, for both LLaVA 1.5 7B and Qwen3VL 4B, where we also consider batch inference (batch size = 4) for Qwen3VL.



> Coverage of competitive baselines: Could you justify the limited coverage of recent learned visual token compression methods? Which representative methods are most directly comparable, and does existing evidence support the reported accuracy-efficiency advantage when model checkpoints, token budgets, training data, adaptation budgets, and evaluation conditions are aligned?

You can feel free to refer to our General Comments "A broader coverage of training-free and training-based baselines". For training-free baselines, in addition to *prompt-agnostic* PruMerge and PruneSID, we also conduct experiments with one *prompt-aware* training-free baseline, SparseVLM. We find that our VLVLM consistently outperforms SparseVLM method, where we observe significant gains under aggressive compression budget (8x). For training-based baselines, other than our default 1D fixed pooling, we extend our experiments on spatially local 2D pooling. The results demonstrate that 2D fixed pooling performs similarly as 1D fixed pooling, and our VLVLM maintains the best OCR-heavy task performance. 

[TOKENPACKER?]


Thank you again for your effort to review our work. We hope you could spend some time reviewing our rebuttal, and we are always happy to further clarify anything needed. If our rebuttal can address your concerns, we would appreciate it if you may reflect that in an updated score.

-------------------------------- the end of responses to u9A8 --------------------------------

