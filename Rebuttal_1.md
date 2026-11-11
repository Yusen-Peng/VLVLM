## for reviewer yzVq

We thanks for reviewer ``yzVq`` feedback and comments. Here is how we address the raised weaknesses and concerns:

> My first concern is the treatment of the final chunk in Algorithms 2–3 of Appendix B. Algorithm 2 takes the K highest-probability boundaries without requiring the last token to be a boundary. Algorithm 3 allocates segments from the boundary count and retains only indices 0 through S−1. For a one-image batch with boundaries [0,1,0,0], the shifted cumulative indices are [0,0,1,1], but S=1, leaving the final two tokens outside every output segment. The implementation may force a terminal boundary or handle the remainder separately, but this is not stated. Please provide implementation-faithful pseudocode that guarantees both full coverage and the exact budget, and specify how soft and hard boundaries connect to the pooling loss through the straight-through estimator.

For the terminal boundary, in order to match the exact budget, we actually allow the last segment to be dropped if the very last image token is not being selected as a boundary and it empirically affects little performance:

```python
# top-k routing: select exactly k boundaries per sample
_, topk_idx = torch.topk(soft_boundaries, k=k, dim=-1)
hard_boundaries = torch.zeros_like(soft_boundaries)
hard_boundaries.scatter_(dim=-1, index=topk_idx, value=1.0)
```

Besides, in our code implementation, we emphasize that the straight-through estimator is only used during training, where we do:

```python
hard_boundaries = (soft_boundaries > 0.5).float()
hard_boundaries = hard_boundaries - soft_boundaries.detach() + soft_boundaries
```

Note that after thresholding during training, we eventually obtain the hard boundaries via the standard straight-through estimator in order to train VLVLM with non-differentiable operations (i.e., boundary selection).


> In my view, Table 5’s text coverage indicates where chunk endpoints fall, not directly how much text information survives pooling. VLVLM averages non-boundary tokens too; an endpoint on text does not guarantee that the corresponding chunk avoids substantial background mixing. The paper itself finds that PruneSID sometimes has higher coverage but worse task performance. I would like chunk-length and text/background-mixing statistics for text-containing chunks, plus trained random variable-length boundaries or spatially local 2D pooling controls. Appendix E already acknowledges vertical-text failures; checking row boundaries and scan order would make the semantic-boundary interpretation more convincing.

Although our proposed VLVLM leads to background mixing due to the nature of token pooling, many more boundary tokens still end up being predicted around its regions, preserving its strong OCR-related task performance. Additionally, mixing the background is *not necessarily a weakness*: depending on the question prompt, it will be beneficial not to discard the background completely.

For spatially local 2D pooling controls, we also conduct experiments on training-based spatially local 2D pooling and here are the results on LLaVA 1.5:

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
| VLVLM (ours) | 55.05 | 15.6 | 20.1| 17.59 | 10.94 | 84.59% |

The results demonstrate that 2D fixed pooling performs similarly as 1D fixed pooling, and our VLVLM maintains the best OCR-heavy task performance.


> The fixed-pooling control is valuable, but PruMerge and PruneSID are still evaluated only as training-free methods. Since training–inference mismatch is central to the motivation, adapting those selectors with the same data and update budget is an especially relevant control. Perceiver does not fully resolve this issue because it changes the compression mechanism as well. LearnPruner is discussed but not directly compared. I would like at least one nonuniform selection or merging baseline with matched fine-tuning, to establish whether the advantage comes from variable-length consecutive grouping rather than primarily from adapting the language model to compressed inputs.


[WAIT-ON-TOKENPACKER?]


> The FLOPs, KV-cache, and memory results in Appendix C are useful, but Table 8’s 55-to-30-hour comparison mainly measures training with versus without FlashAttention, not inference acceleration over baselines using the same backend. Algorithm 3 also constructs a token-to-segment assignment matrix, and it is unclear whether the implementation uses a more efficient segmented reduction. Please report time to first token, end-to-end latency at fixed generation lengths, and the predictor/pooling overhead under matched settings, particularly for high-resolution Qwen3VL. This would establish whether the arithmetic savings translate into deployment benefits.


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

Under 4x compression, most methods and baselines including VLVLM can reach up to **2x** prefill latency speedup with a single batch size; SparseVLM only reaches up to **1.5x** latency boost since it starts merging image tokens during the LLM layer. Note that for LLaVA, all images are resized into the 336x336 resolution, leading to a constant number of image tokens to start with (576 with patch size 14).


To validate the latency gains with more modern VLMs, we further measure the prefill latency improvement on multiple benchmarks including OCRBench, TextVQA, and ChartQAPro for Qwen3VL 4B model. We first conduct a profiling experiment on a single-sample (batch size = 1) setting, and VLVLM achieves similar prefill speedup as other baselines such as fixed pooling and PruneSID. To maximize GPU utilization, we also consider batched (batch size = 4) setting for inference, which is not supported by PruneSID.

| Method | OCRBench | TextVQA | ChartQAPro | OCRBench | TextVQA | ChartQAPro |
| ------ | -------- | ------- | ---------- | -------- | ------- | ---------- |
| batch size = 1 | | | | batch size = 4 |
| Qwen3VL-4B | 92.02 | 69.50 | 131.79 | 406.78 | 299.09 | 830.45 |
| 4x compression |
| PruneSID | 57.77 | 53.54 | 63.62 | -- | -- | -- |
| Fixed pooling | 57.76 | 53.87 | 63.49 | 123.85 | 147.11 | 216.54 |
| VLVLM | 58.45 | 54.00 | 63.02 | 124.36 | 143.52 | 215.41 |
