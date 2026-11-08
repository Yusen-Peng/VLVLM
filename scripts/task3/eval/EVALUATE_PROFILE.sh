#!/bin/bash

# # MME
sbatch scripts/task3/eval/eval_MME.sh

# Reasoning (2): GQA
sbatch scripts/task3/eval/eval_GQA.sh


# OCR (5): TextVQA
sbatch scripts/task3/eval/eval_textVQA.sh
# DocVQA
sbatch scripts/task3/eval/eval_DocVQA.sh


# Hallucination (1): POPE
sbatch scripts/task3/eval/eval_POPE.sh

# MM-Vet
sbatch scripts/task3/eval/eval_MMVet.sh
