#!/bin/bash
#SBATCH --job-name=1028_TextVQA_Qwen3VL_SFT_Fixed_profile
#SBATCH --output=1028_TextVQA_Qwen3VL_SFT_Fixed_profile.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP_qwenvl_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

VERSION="1028_TextVQA_Qwen3VL_SFT_Fixed_profile"

QUESTION_FILE="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_FOLDER="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/train_images"
ANSWER_FOLDER="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers"

mkdir -p "${ANSWER_FOLDER}"

# ============================================================
# Original Qwen3-VL SFT
# ============================================================

# python eval_code/model_vqa_loader_profile.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${QUESTION_FILE}" \
#     --image-folder "${IMAGE_FOLDER}" \
#     --answers-file "${ANSWER_FOLDER}/${VERSION}_original_latency.jsonl" \
#     --temperature 0 \
#     --merge-strategy none \
#     --measure-latency

# ============================================================
# PruneSID 4x
# ============================================================

# python eval_code/model_vqa_loader_profile.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${QUESTION_FILE}" \
#     --image-folder "${IMAGE_FOLDER}" \
#     --answers-file "${ANSWER_FOLDER}/${VERSION}_PruneSID_4x_latency.jsonl" \
#     --temperature 0 \
#     --merge-strategy PruneSID \
#     --compression-rate 0.25 \
#     --measure-latency

python eval_code/model_vqa_loader_profile.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
    --model-base Qwen/Qwen3-VL-4B-Instruct \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_FOLDER}/${VERSION}_PruneSID_4x_latency.jsonl" \
    --temperature 0 \
    --merge-strategy Fixed \
    --compression-rate 0.25 \
    --measure-latency


# ============================================================
# VLVLM (DRIP) 4x
# ============================================================

# python eval_code/model_vqa_loader_profile.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_Distill_rerun \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${QUESTION_FILE}" \
#     --image-folder "${IMAGE_FOLDER}" \
#     --answers-file "${ANSWER_FOLDER}/${VERSION}_VLVLM_4x_latency.jsonl" \
#     --temperature 0 \
#     --merge-strategy DRIP \
#     --compression-rate 0.25 \
#     --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_Distill_rerun/drip.bin \
#     --mlp-ratio 4.0 \
#     --measure-latency


# ============================================================
# TextVQA Evaluation
# ============================================================

cd /users/PAS2912/yusenpeng/DRIP

python src/eval_textvqa.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/TextVQA_0.5.1_val.json \
    --result-file "${ANSWER_FOLDER}/${VERSION}_VLVLM_4x_latency.jsonl"

conda deactivate
