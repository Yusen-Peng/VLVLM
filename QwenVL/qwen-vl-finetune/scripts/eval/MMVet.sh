#!/bin/bash
#SBATCH --job-name=1109_MMVet_Qwen3VL_VLVLM
#SBATCH --output=1109_MMVet_Qwen3VL_VLVLM.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

set -e

module load miniconda3/24.1.2-py310
conda activate DRIP_qwenvl_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))
export CUDA_LAUNCH_BLOCKING=1

VERSION="1109_MMVet_Qwen3VL_VLVLM"

MMVET_DIR="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet"
OUTPUT_DIR="${MMVET_DIR}/answers"
OUTPUT_FILE="${OUTPUT_DIR}/${VERSION}.jsonl"

mkdir -p "${OUTPUT_DIR}"

cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

# ============================================================
# Qwen3VL Baseline (No Fine-tuning)
# ============================================================
# python eval_code/model_vqa_loader.py \
#     --model-path Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${MMVET_DIR}/llava-mm-vet.jsonl" \
#     --image-folder "${MMVET_DIR}/images" \
#     --answers-file "${OUTPUT_FILE}" \
#     --temperature 0


# ============================================================
# Qwen3VL SFT (No Compression)
# ============================================================
# python eval_code/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${MMVET_DIR}/llava-mm-vet.jsonl" \
#     --image-folder "${MMVET_DIR}/images" \
#     --answers-file "${OUTPUT_FILE}" \
#     --temperature 0


# ============================================================
# Qwen3VL SFT + PruneSID (4x)
# ============================================================
# python eval_code/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${MMVET_DIR}/llava-mm-vet.jsonl" \
#     --image-folder "${MMVET_DIR}/images" \
#     --answers-file "${OUTPUT_FILE}" \
#     --temperature 0 \
#     --merge-strategy PruneSID \
#     --compression-rate 0.25


# ============================================================
# Qwen3VL SFT + Fixed Pooling (4x, Transfer)
# ============================================================
# python eval_code/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_4x_NEW_PIPELINE_transfer \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${MMVET_DIR}/llava-mm-vet.jsonl" \
#     --image-folder "${MMVET_DIR}/images" \
#     --answers-file "${OUTPUT_FILE}" \
#     --temperature 0 \
#     --merge-strategy Fixed \
#     --compression-rate 0.25


# ============================================================
# Qwen3VL SFT + DRIP (4x, Transfer)
# ============================================================
python eval_code/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_transfer \
    --model-base Qwen/Qwen3-VL-4B-Instruct \
    --question-file "${MMVET_DIR}/llava-mm-vet.jsonl" \
    --image-folder "${MMVET_DIR}/images" \
    --answers-file "${OUTPUT_FILE}" \
    --temperature 0 \
    --merge-strategy DRIP \
    --compression-rate 0.25 \
    --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_transfer/drip.bin \
    --mlp-ratio 4.0


# ============================================================
# MM-Vet Answer Conversion
# ============================================================

cd /users/PAS2912/yusenpeng/DRIP/

mkdir -p "${MMVET_DIR}/results"

python src/convert_mmvet_for_eval.py \
    --src "${OUTPUT_FILE}" \
    --dst "${MMVET_DIR}/results/${VERSION}.json"


# ============================================================
# MM-Vet Evaluation
# ============================================================

echo "✅ MM-Vet inference and conversion completed: ${VERSION}"
echo "Result file: ${MMVET_DIR}/results/${VERSION}.json"
echo "Submit to: https://huggingface.co/spaces/whyu/MM-Vet_Evaluator"

conda deactivate
