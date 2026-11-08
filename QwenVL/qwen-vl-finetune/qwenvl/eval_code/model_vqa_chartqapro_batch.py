
import argparse
import json
import os
import sys
import statistics

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import shortuuid


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../"))
sys.path.insert(0, PROJECT_ROOT)

from eval_code.model_vqa_loader_batch import (
    load_model,
    get_chunk,
)
from eval_code.latency_profiler import Qwen3VLLatencyProfiler


# ============================================================
# Dataset
# ============================================================

class ChartQAProDataset(Dataset):

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

        # Preserve original ChartQAPro evaluation prompt.
        question = (
            line["text"]
            + "\nAnswer the question using a single word or phrase."
        )

        return {
            "image": image,
            "question": question,
            "line": line,
        }


# ============================================================
# Batched processor
# ============================================================

class ChartQAProCollator:

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

        return inputs, lines


def create_data_loader(
    questions,
    image_folder,
    processor,
    batch_size=1,
    num_workers=0,
):
    dataset = ChartQAProDataset(
        questions,
        image_folder,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ChartQAProCollator(processor),
    )


# ============================================================
# Generation arguments
# ============================================================

def get_generation_kwargs(args):

    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "use_cache": True,
    }

    if args.temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = args.temperature

        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
    else:
        kwargs["do_sample"] = False

    return kwargs


# ============================================================
# Evaluation
# ============================================================

def eval_model(args):

    model, processor, model_name = load_model(args)

    profiler = None
    warmup_done = False
    latencies = []

    if args.measure_latency:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for latency profiling."
            )

        profiler = Qwen3VLLatencyProfiler(model)
        profiler.install()

    # --------------------------------------------------------
    # Load ChartQAPro questions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    answers_file = os.path.expanduser(args.answers_file)

    answer_dir = os.path.dirname(answers_file)

    if answer_dir:
        os.makedirs(answer_dir, exist_ok=True)

    latency_file = answers_file + ".latency.jsonl"

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    data_loader = create_data_loader(
        questions=questions,
        image_folder=os.path.expanduser(args.image_folder),
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    generation_kwargs = get_generation_kwargs(args)

    print(f"🚀 ChartQAPro batch size: {args.batch_size}")
    print(f"🚀 Number of questions: {len(questions)}")
    print(f"⏱️ Profiling: {args.measure_latency}")

    lat_file = None

    try:
        with open(answers_file, "w") as ans_file:

            if args.measure_latency:
                lat_file = open(latency_file, "w")

            for inputs, lines in tqdm(
                data_loader,
                total=len(data_loader),
            ):

                actual_batch_size = len(lines)

                # --------------------------------------------
                # Validate multimodal batch
                # --------------------------------------------

                if inputs["input_ids"].shape[0] != actual_batch_size:
                    raise RuntimeError(
                        "Processor batch size mismatch."
                    )

                if "image_grid_thw" not in inputs:
                    raise RuntimeError(
                        "Missing image_grid_thw."
                    )

                if inputs["image_grid_thw"].shape[0] != actual_batch_size:
                    raise RuntimeError(
                        "Expected exactly one image per sample."
                    )

                inputs = inputs.to(model.device)

                prompt_length = inputs["input_ids"].shape[1]

                # --------------------------------------------
                # Warmup (once, outside timed measurements)
                # --------------------------------------------

                if args.measure_latency and not warmup_done:

                    print(
                        f"🔥 Warming up with batch size "
                        f"{actual_batch_size}"
                    )

                    with torch.inference_mode():
                        for _ in range(args.latency_warmup):
                            model.generate(
                                **inputs,
                                **generation_kwargs,
                            )

                    torch.cuda.synchronize()
                    warmup_done = True

                # --------------------------------------------
                # Generation + optional profiling
                # --------------------------------------------

                if args.measure_latency:

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

                    latency = dict(latency)

                    generated_tokens = int(
                        generated_ids.shape[1] - prompt_length
                    )

                    prefill_ms = latency.get("prefill_ms")

                    latency.update({
                        "question_ids": [
                            line["question_id"]
                            for line in lines
                        ],
                        "batch_size": actual_batch_size,
                        "prompt_length_padded": prompt_length,
                        "generated_tokens": generated_tokens,
                        "generated_token_slots": (
                            generated_tokens * actual_batch_size
                        ),
                        "prefill_throughput_img_s": (
                            actual_batch_size * 1000.0 / prefill_ms
                            if prefill_ms is not None and prefill_ms > 0
                            else None
                        ),
                        "peak_memory_gib": (
                            torch.cuda.max_memory_allocated()
                            / (1024 ** 3)
                        ),
                    })

                    latencies.append(latency)

                    lat_file.write(
                        json.dumps(latency) + "\n"
                    )
                    lat_file.flush()

                else:

                    with torch.inference_mode():
                        generated_ids = model.generate(
                            **inputs,
                            **generation_kwargs,
                        )

                # --------------------------------------------
                # Decode all samples
                # --------------------------------------------

                generated_ids_trimmed = generated_ids[
                    :, prompt_length:
                ]

                outputs = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                if len(outputs) != actual_batch_size:
                    raise RuntimeError(
                        "Generated output count mismatch."
                    )

                # --------------------------------------------
                # Preserve original output JSONL schema
                # --------------------------------------------

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

    finally:
        if lat_file is not None:
            lat_file.close()

        if profiler is not None:
            profiler.remove()

    # ========================================================
    # Latency summary
    # ========================================================

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
            x["batch_size"] for x in latencies
        )

        total_prefill_ms = sum(
            x["prefill_ms"]
            for x in latencies
            if x.get("prefill_ms") is not None
        )

        total_decode_ms = sum(
            x["decode_ms"] for x in latencies
        )

        total_decode_steps = sum(
            x["decode_steps"] for x in latencies
        )

        total_forward_ms = sum(
            x["total_forward_ms"]
            for x in latencies
        )

        summary = {
            "model": model_name,
            "merge_strategy": args.merge_strategy,
            "compression_rate": args.compression_rate,
            "batch_size": args.batch_size,
            "num_samples": total_samples,
            "num_batches": len(latencies),
            "num_chunks": args.num_chunks,
            "chunk_idx": args.chunk_idx,

            # Latency (ms per batch)
            "prefill_ms": avg("prefill_ms"),
            "llm_prefill_ms": avg("llm_prefill_ms"),
            "decode_tpot_ms": (
                total_decode_ms / total_decode_steps
                if total_decode_steps else None
            ),
            "decode_total_ms": avg("decode_ms"),
            "total_forward_ms": avg("total_forward_ms"),

            # Throughput
            "prefill_throughput_img_s": (
                total_samples * 1000.0 / total_prefill_ms
                if total_prefill_ms > 0 else None
            ),
            "total_forward_throughput_img_s": (
                total_samples * 1000.0 / total_forward_ms
                if total_forward_ms > 0 else None
            ),

            # GPU peak allocated memory
            "peak_memory_gib": max(
                x["peak_memory_gib"]
                for x in latencies
            ),

            # Generation
            "mean_generated_tokens": avg("generated_tokens"),
            "mean_decode_steps": avg("decode_steps"),
        }

        summary_path = (
            answers_file + ".latency_summary.json"
        )

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print("\n" + "=" * 60)
        print("⏱️ Qwen3-VL ChartQAPro Batched Latency Summary")
        print("=" * 60)

        for key, value in summary.items():
            print(f"{key}: {value}")

        print("=" * 60)


# ============================================================
# CLI
# ============================================================

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
        "--measure-latency",
        action="store_true",
    )

    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")

    if args.merge_strategy in {"Fixed", "DRIP"}:
        if args.num_beams != 1:
            parser.error(
                "Batched compressed inference requires --num_beams 1"
            )

    eval_model(args)
