#!/bin/bash
#SBATCH --job-name=1110_MMBench_Qwen3VL
#SBATCH --output=1110_MMBench_Qwen3VL.log
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
conda activate DRIP_qwenvl_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))
export CUDA_LAUNCH_BLOCKING=1

VERSION="1110_MMBench_Qwen3VL"

MMBENCH_DIR="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench"

QUESTION_FILE="${MMBENCH_DIR}/mmbench_dev_20230712.tsv"
OUTPUT_DIR="${MMBENCH_DIR}/answers/mmbench_dev_20230712"
OUTPUT_FILE="${OUTPUT_DIR}/${VERSION}.jsonl"

UPLOAD_DIR="${MMBENCH_DIR}/answers_upload/mmbench_dev_20230712"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${UPLOAD_DIR}"

cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

# ============================================================
# Qwen3VL Baseline (No Fine-tuning)
# ============================================================
python eval_code/model_vqa_mmbench.py \
    --model-path Qwen/Qwen3-VL-4B-Instruct \
    --question-file "${QUESTION_FILE}" \
    --answers-file "${OUTPUT_FILE}" \
    --single-pred-prompt \
    --temperature 0


# ============================================================
# Qwen3VL SFT (No Compression)
# ============================================================
# python eval_code/model_vqa_mmbench.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${QUESTION_FILE}" \
#     --answers-file "${OUTPUT_FILE}" \
#     --single-pred-prompt \
#     --temperature 0


# ============================================================
# Qwen3VL SFT + PruneSID (4x)
# ============================================================
# python eval_code/model_vqa_mmbench.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${QUESTION_FILE}" \
#     --answers-file "${OUTPUT_FILE}" \
#     --single-pred-prompt \
#     --temperature 0 \
#     --merge-strategy PruneSID \
#     --compression-rate 0.25



# ============================================================
# Qwen3VL SFT + Fixed Pooling (4x, Transfer)
# ============================================================
# python eval_code/model_vqa_mmbench.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_4x_NEW_PIPELINE_transfer \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${QUESTION_FILE}" \
#     --answers-file "${OUTPUT_FILE}" \
#     --single-pred-prompt \
#     --temperature 0 \
#     --merge-strategy Fixed \
#     --compression-rate 0.25


# ============================================================
# Qwen3VL SFT + DRIP (4x, Transfer)
# ============================================================
# python eval_code/model_vqa_mmbench.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_transfer \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file "${QUESTION_FILE}" \
#     --answers-file "${OUTPUT_FILE}" \
#     --single-pred-prompt \
#     --temperature 0 \
#     --merge-strategy DRIP \
#     --compression-rate 0.25 \
#     --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_transfer/drip.bin \
#     --mlp-ratio 4.0


# ============================================================
# MMBench Submission Conversion
# ============================================================

cd /users/PAS2912/yusenpeng/DRIP/

python src/convert_mmbench_for_submission.py \
    --annotation-file "${QUESTION_FILE}" \
    --result-dir "${OUTPUT_DIR}" \
    --upload-dir "${UPLOAD_DIR}" \
    --experiment "${VERSION}"

echo "✅ MMBench inference and conversion completed: ${VERSION}"
echo "Submission directory: ${UPLOAD_DIR}"

conda deactivate
