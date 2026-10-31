#!/bin/bash
#SBATCH --job-name=1028_OCRBench_Qwen3VL_SFT_Distill_DRIP_4x
#SBATCH --output=1028_OCRBench_Qwen3VL_SFT_Distill_DRIP_4x.log
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

VERSION="Qwen3VL_SFT_Distill_DRIP_4x"

# python eval_code/model_vqa_ocrbench.py \
#     --model_path Qwen/Qwen3-VL-4B-Instruct \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --num_workers 1


# python eval_code/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model_base Qwen/Qwen3-VL-4B-Instruct \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --num_workers 1

# python eval_code/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model_base Qwen/Qwen3-VL-4B-Instruct \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --num_workers 1 \
#     --merge-strategy CDPruner \
#     --compression-rate 0.25



# python eval_code/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Distill \
#     --model_base Qwen/Qwen3-VL-4B-Instruct \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --num_workers 1 \
#     --merge-strategy CDPruner \
#     --compression-rate 0.25


# python eval_code/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_4x_Distill \
#     --model_base Qwen/Qwen3-VL-4B-Instruct \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --num_workers 1 \
#     --merge-strategy Fixed \
#     --compression-rate 0.25

python eval_code/model_vqa_ocrbench.py \
    --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_Distill \
    --model_base Qwen/Qwen3-VL-4B-Instruct \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
    --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
    --save_name ${VERSION} \
    --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
    --temperature 0 \
    --num_workers 1 \
    --merge-strategy DRIP \
    --compression-rate 0.25 \
    --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_Distill/drip.bin \
    --mlp-ratio 4.0


# python eval_code/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_4x_NEW_PIPELINE_transfer \
#     --model_base Qwen/Qwen3-VL-4B-Instruct \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --num_workers 1 \
#     --merge-strategy Fixed \
#     --compression-rate 0.25


# python eval_code/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_transfer \
#     --model_base Qwen/Qwen3-VL-4B-Instruct \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --num_workers 1 \
#     --merge-strategy DRIP \
#     --compression-rate 0.25 \
#     --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_transfer/drip.bin \
#     --mlp-ratio 4.0



conda deactivate
