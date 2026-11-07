#!/bin/bash
#SBATCH --job-name=1028_ChartQAPro_Qwen3VL_PruneSID_profile
#SBATCH --output=1028_ChartQAPro_Qwen3VL_PruneSID_profile.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

set -e
module load miniconda3/24.1.2-py310
module load cuda/12.6.2
conda activate DRIP_qwenvl_flash

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

VERSION="1028_ChartQAPro_Qwen3VL_PruneSID_profile"
QUESTION_FILE="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl"
IMAGE_FOLDER="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/images"
RESULT_DIR="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results"
mkdir -p "${RESULT_DIR}"
ANSWERS_FILE="${RESULT_DIR}/${VERSION}_VLVLM_4x_latency.jsonl"

# ============================================================
# Original Qwen3-VL SFT (uncomment to run instead)
# ============================================================
# ANSWERS_FILE="${RESULT_DIR}/${VERSION}_original_latency.jsonl"
# python eval_code/model_vqa_chartqapro_profile.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#   --model-base Qwen/Qwen3-VL-4B-Instruct \
#   --image-folder "${IMAGE_FOLDER}" \
#   --question-file "${QUESTION_FILE}" \
#   --answers-file "${ANSWERS_FILE}" \
#   --temperature 0 \
#   --conv-mode llava_v1 \
#   --merge-strategy none \
#   --measure-latency

# ============================================================
# PruneSID 4x (uncomment to run instead)
# ============================================================
ANSWERS_FILE="${RESULT_DIR}/${VERSION}_PruneSID_4x_latency.jsonl"
python eval_code/model_vqa_chartqapro_profile.py \
  --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
  --model-base Qwen/Qwen3-VL-4B-Instruct \
  --image-folder "${IMAGE_FOLDER}" \
  --question-file "${QUESTION_FILE}" \
  --answers-file "${ANSWERS_FILE}" \
  --temperature 0 \
  --conv-mode llava_v1 \
  --merge-strategy PruneSID \
  --compression-rate 0.25 \
  --measure-latency

# ============================================================
# VLVLM (DRIP) 4x — active
# ============================================================
# python eval_code/model_vqa_chartqapro_profile.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_Distill_rerun \
#   --model-base Qwen/Qwen3-VL-4B-Instruct \
#   --image-folder "${IMAGE_FOLDER}" \
#   --question-file "${QUESTION_FILE}" \
#   --answers-file "${ANSWERS_FILE}" \
#   --temperature 0 \
#   --conv-mode llava_v1 \
#   --merge-strategy DRIP \
#   --compression-rate 0.25 \
#   --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_Distill_rerun/drip.bin \
#   --mlp-ratio 4.0 \
#   --measure-latency


# ============================================================
# ChartQAPro scoring
# ============================================================
cd /users/PAS2912/yusenpeng/DRIP
python src/ChartQAPro_eval/eval_prediction.py \
  --predictions-file "${ANSWERS_FILE}" \
  --gt-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_gt.json

conda deactivate
