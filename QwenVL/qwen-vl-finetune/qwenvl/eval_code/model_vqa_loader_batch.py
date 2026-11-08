
import argparse
import json
import math
import os
import sys
import statistics

import shortuuid
import torch

from PIL import Image
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../"))
sys.path.insert(0, PROJECT_ROOT)

from model.qwen3vl_compressed_batch import BatchedQwen3VLForConditionalGeneration
from eval_code.latency_profiler import Qwen3VLLatencyProfiler


SUPPORTED_STRATEGIES = {"none", "DRIP", "Fixed"}


# ============================================================
# Question chunking (UNCHANGED)
# ============================================================

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [
        lst[i:i + chunk_size]
        for i in range(0, len(lst), chunk_size)
    ]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


# ============================================================
# Dataset
# ============================================================

class CustomDataset(Dataset):

    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        line = self.questions[index]

        image_path = os.path.join(
            self.image_folder,
            line["image"],
        )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        return {
            "image": image,
            "question": line["text"],
            "line": line,
        }


# ============================================================
# Batched Qwen3-VL preprocessing
# ============================================================

class BatchedCollator:
    """
    The processor receives a batch of conversations.

    It handles:
        - chat templates
        - tokenization
        - left/right padding according to tokenizer config
        - image preprocessing
        - image_grid_thw creation

    One image per question.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        conversations = []

        for item in batch:
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": item["image"],
                        },
                        {
                            "type": "text",
                            "text": item["question"],
                        },
                    ],
                }
            ])

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )

        lines = [item["line"] for item in batch]
        image_sizes = [item["image"].size for item in batch]

        return inputs, lines, image_sizes


def create_data_loader(
    questions,
    image_folder,
    processor,
    batch_size=1,
    num_workers=0,
):
    dataset = CustomDataset(
        questions=questions,
        image_folder=image_folder,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=BatchedCollator(processor),
    )

    return data_loader


# ============================================================
# Model loading
# ============================================================

def load_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_base = (
        os.path.expanduser(args.model_base)
        if args.model_base is not None
        else None
    )

    if args.merge_strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported merge strategy: {args.merge_strategy}. "
            f"Supported: {sorted(SUPPORTED_STRATEGIES)}"
        )

    # Match the original loader's inference configuration.
    load_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    }

    checkpoint_source = (
        model_base if model_base is not None else model_path
    )

    # --------------------------------------------------------
    # Compressed architecture: DRIP / Fixed
    # --------------------------------------------------------

    if args.merge_strategy in {"DRIP", "Fixed"}:
        print(
            f"🌊 Loading batched compressed Qwen3-VL: "
            f"{args.merge_strategy}, rate={args.compression_rate}"
        )

        model = (
            BatchedQwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_source,
                **load_kwargs,
            )
        )

        model.model.set_compressor(
            merge_strategy=args.merge_strategy,
            compression_rate=args.compression_rate,
            temperature=args.sampling_temperature,
            drip_path=args.drip_path,
            mlp_ratio=args.mlp_ratio,
        )


    else:
        print("🌊 Loading original Qwen3-VL (batched baseline)")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_source,
            **load_kwargs,
        )

    if model_base is not None:
        print(f"Loading LoRA adapter from {model_path}")

        model = PeftModel.from_pretrained(
            model,
            model_path,
        )

        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(
        checkpoint_source
    )

    # Critical for batched decoder-only generation.
    processor.tokenizer.padding_side = "left"

    # Some Qwen tokenizers do not define a separate pad token.
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    model_name = os.path.basename(
        os.path.normpath(model_path)
    )

    model.eval()

    return model, processor, model_name


def eval_model(args):

    model, processor, model_name = load_model(args)

    profiler = None
    warmup_done = False
    latencies = []
    if args.measure_latency:
        profiler = Qwen3VLLatencyProfiler(model)
        profiler.install()


    with open(
        os.path.expanduser(args.question_file), "r"
    ) as f:
        questions = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    questions = get_chunk(
        questions,
        args.num_chunks,
        args.chunk_idx,
    )

    answers_file = os.path.expanduser(args.answers_file)

    answer_dir = os.path.dirname(answers_file)

    if answer_dir:
        os.makedirs(answer_dir, exist_ok=True)


    latency_file = answers_file + ".latency.jsonl"

    lat_file = (
        open(latency_file, "w")
        if args.measure_latency
        else None
    )


    data_loader = create_data_loader(
        questions=questions,
        image_folder=os.path.expanduser(args.image_folder),
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "use_cache": True,
    }

    if args.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature

        if args.top_p is not None:
            generation_kwargs["top_p"] = args.top_p
    else:
        generation_kwargs["do_sample"] = False

    print(f"🚀 Batch size: {args.batch_size}")
    print(f"🚀 Number of questions: {len(questions)}")

    with open(answers_file, "w") as ans_file:

        for inputs, lines, _ in tqdm(data_loader, total=len(data_loader)):
            actual_batch_size = len(lines)

            # Processor produces flattened image patches
            # and one grid_thw entry per image.
            if "image_grid_thw" not in inputs:
                raise RuntimeError("Qwen processor did not return image_grid_thw.")

            if inputs["image_grid_thw"].shape[0] != actual_batch_size:
                raise RuntimeError(
                    "Expected exactly one image per question. "
                    f"Batch={actual_batch_size}, "
                    f"images={inputs['image_grid_thw'].shape[0]}"
                )

            if inputs["input_ids"].shape[0] != actual_batch_size:
                raise RuntimeError("Processor batch size mismatch.")

            inputs = inputs.to(model.device)

            # One shared padded prompt length.
            # generate() returns that full padded prefix followed by generated tokens.
            prompt_length = inputs["input_ids"].shape[1]

            if args.measure_latency:

                # ========================================================
                # Warmup: first batch only, outside timed measurements
                # ========================================================

                if not warmup_done:
                    print(f"🔥 Warming up with batch size {actual_batch_size}")

                    with torch.inference_mode():
                        for _ in range(args.latency_warmup):
                            model.generate(
                                **inputs,
                                **generation_kwargs,
                            )

                    torch.cuda.synchronize()
                    warmup_done = True

                # ========================================================
                # Measure CUDA latency and peak allocated memory
                # ========================================================

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                profiler.start()

                try:
                    with torch.inference_mode():
                        generated_ids = model.generate(
                            **inputs,
                            **generation_kwargs,
                        )

                    latency = profiler.stop()
                    torch.cuda.synchronize()

                except Exception:
                    profiler.enabled = False
                    raise

                # ========================================================
                # Batch-level metadata
                # ========================================================

                generated_tokens_per_sample = int(
                    generated_ids.shape[1] - prompt_length
                )

                latency["question_ids"] = [
                    line["question_id"]
                    for line in lines
                ]

                latency["batch_size"] = actual_batch_size

                latency["generated_tokens"] = (
                    generated_tokens_per_sample
                )

                # Counts generated token slots, including padding
                # for sequences that terminate earlier than others.
                latency["generated_token_slots"] = (
                    generated_tokens_per_sample * actual_batch_size
                )

                latency["prompt_length_padded"] = prompt_length

                latency["prefill_throughput_img_s"] = (
                    actual_batch_size * 1000.0 / latency["prefill_ms"]
                    if latency.get("prefill_ms", 0) > 0
                    else None
                )

                latency["peak_memory_gib"] = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                )

                latencies.append(latency)

                lat_file.write(json.dumps(latency) + "\n")
                lat_file.flush()

            else:

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )

            # Trim the padded original prompt length.
            generated_ids_trimmed = generated_ids[:, prompt_length:]

            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if len(outputs) != actual_batch_size:
                raise RuntimeError(f"Generated {len(outputs)} outputs for {actual_batch_size} questions.")

            for line, output in zip(lines, outputs):
                ans_file.write(
                    json.dumps({
                        "question_id": line["question_id"],
                        "prompt": line["text"],
                        "text": output.strip(),
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {},
                    }) + "\n"
                )
            ans_file.flush()

    # ============================================================
    # Finalize profiling
    # ============================================================

    if lat_file is not None:
        lat_file.close()
        profiler.remove()


    if args.measure_latency and latencies:

        def avg(key):
            values = [
                record[key]
                for record in latencies
                if record.get(key) is not None
            ]

            return (
                float(statistics.mean(values))
                if values
                else None
            )

        total_samples = sum(
            record["batch_size"]
            for record in latencies
        )

        total_prefill_ms = sum(
            record["prefill_ms"]
            for record in latencies
            if record.get("prefill_ms") is not None
        )

        total_decode_ms = sum(
            record["decode_ms"]
            for record in latencies
        )

        total_decode_steps = sum(
            record["decode_steps"]
            for record in latencies
        )

        total_forward_ms = sum(
            record["total_forward_ms"]
            for record in latencies
        )

        # ========================================================
        # Throughput metrics
        # ========================================================

        prefill_throughput = (
            total_samples * 1000.0 / total_prefill_ms
            if total_prefill_ms > 0
            else None
        )

        decode_tpot_ms = (
            total_decode_ms / total_decode_steps
            if total_decode_steps > 0
            else None
        )

        total_forward_throughput = (
            total_samples * 1000.0 / total_forward_ms
            if total_forward_ms > 0
            else None
        )

        # ========================================================
        # Summary
        # ========================================================

        summary = {
            "model": model_name,
            "merge_strategy": args.merge_strategy,
            "compression_rate": args.compression_rate,

            "batch_size": args.batch_size,
            "num_samples": total_samples,
            "num_batches": len(latencies),

            "num_chunks": args.num_chunks,
            "chunk_idx": args.chunk_idx,

            # Latency: milliseconds per batch
            "prefill_ms": avg("prefill_ms"),
            "llm_prefill_ms": avg("llm_prefill_ms"),
            "decode_tpot_ms": decode_tpot_ms,
            "decode_total_ms": avg("decode_ms"),
            "total_forward_ms": avg("total_forward_ms"),

            # Throughput: images/s
            "prefill_throughput_img_s": prefill_throughput,
            "total_forward_throughput_img_s": (
                total_forward_throughput
            ),

            # Memory: maximum allocated GiB
            "peak_memory_gib": max(
                record["peak_memory_gib"]
                for record in latencies
            ),

            # Generation statistics
            "mean_generated_tokens": avg("generated_tokens"),
            "mean_decode_steps": avg("decode_steps"),
        }

        summary_path = (
            answers_file + ".latency_summary.json"
        )

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print("\n" + "=" * 60)
        print("⏱️ Qwen3-VL Batched Latency Summary")
        print("=" * 60)

        for key, value in summary.items():
            print(f"{key}: {value}")
        print("=" * 60)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
    )

    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--image-folder",
        type=str,
        default="",
    )

    parser.add_argument(
        "--question-file",
        type=str,
        default="tables/question.jsonl",
    )

    parser.add_argument(
        "--answers-file",
        type=str,
        default="answer.jsonl",
    )

    parser.add_argument(
        "--conv-mode",
        type=str,
        default="qwen3_vl",
    )

    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--merge-strategy",
        type=str,
        default="none",
    )

    parser.add_argument(
        "--compression-rate",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--drip-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=1.0,
    )

    # NEW arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--measure-latency", action="store_true",
        help="Record prefill and decoding CUDA-event timings",
    )
    parser.add_argument(
        "--latency-warmup", type=int, default=3,
        help="Number of untimed generate() calls on the first sample",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    eval_model(args)
