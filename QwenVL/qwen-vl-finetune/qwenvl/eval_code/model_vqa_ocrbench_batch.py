
import argparse
import json
import os
import sys
import statistics
import multiprocessing
from contextlib import nullcontext

import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../"))
sys.path.insert(0, PROJECT_ROOT)

from model.qwen3vl_compressed_batch import (
    BatchedQwen3VLForConditionalGeneration,
)
from eval_code.latency_profiler import Qwen3VLLatencyProfiler


SUPPORTED_STRATEGIES = {"none", "Fixed", "DRIP"}


# ============================================================
# Utilities
# ============================================================

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def split_list(lst, n):
    """Split into n nearly equal contiguous partitions."""
    q, r = divmod(len(lst), n)
    result = []
    offset = 0

    for i in range(n):
        size = q + (i < r)
        result.append(lst[offset:offset + size])
        offset += size

    return result


def has_prediction(sample):
    return sample.get("predict", 0) not in (0, None, "")


# ============================================================
# OCRBench scoring (same matching rules as original)
# ============================================================

def score_sample(sample):
    if not has_prediction(sample):
        return sample

    dataset_name = sample["dataset_name"]
    answers = sample["answers"]
    predict = sample["predict"]

    if not isinstance(answers, list):
        answers = [answers]

    if dataset_name == "HME100k":
        normalized_predict = (
            predict.strip()
            .replace("\n", " ")
            .replace(" ", "")
        )
        normalized_answers = [
            answer.strip()
            .replace("\n", " ")
            .replace(" ", "")
            for answer in answers
        ]
    else:
        normalized_predict = (
            predict.lower()
            .strip()
            .replace("\n", " ")
        )
        normalized_answers = [
            answer.lower()
            .strip()
            .replace("\n", " ")
            for answer in answers
        ]

    sample["result"] = int(
        any(
            answer in normalized_predict
            for answer in normalized_answers
        )
    )

    return sample


# ============================================================
# Dataset
# ============================================================

class OCRBenchDataset(Dataset):

    def __init__(self, indexed_samples, image_folder):
        self.indexed_samples = indexed_samples
        self.image_folder = image_folder

    def __len__(self):
        return len(self.indexed_samples)

    def __getitem__(self, index):
        original_index, sample = self.indexed_samples[index]

        image_path = os.path.join(
            self.image_folder,
            sample["image_path"],
        )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Preserve the original OCRBench prompt.
        question = (
            sample["question"]
            + "\nAnswer the question using a single word or phrase."
        )

        return {
            "index": original_index,
            "image": image,
            "question": question,
        }


# ============================================================
# Batched processor
# ============================================================

class OCRBenchCollator:

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

        indices = [item["index"] for item in batch]

        return inputs, indices


# ============================================================
# Model loading
# ============================================================

def load_model(args, device):

    model_path = os.path.expanduser(args.model_path)

    model_base = (
        os.path.expanduser(args.model_base)
        if args.model_base is not None
        else None
    )

    checkpoint_source = (
        model_base if model_base is not None else model_path
    )

    load_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": {"": device},
    }

    if args.merge_strategy in {"Fixed", "DRIP"}:

        print(
            f"🌊 [{device}] Loading batched compressed Qwen3-VL: "
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

        print(f"🌊 [{device}] Loading original Qwen3-VL")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_source,
            **load_kwargs,
        )

    # LoRA adapter, if supplied.
    if model_base is not None:

        print(
            f"[{device}] Loading LoRA adapter from {model_path}"
        )

        model = PeftModel.from_pretrained(
            model,
            model_path,
        )

        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(
        checkpoint_source
    )

    processor.tokenizer.padding_side = "left"

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    model.eval()

    return model, processor


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
# Evaluation worker: one GPU, genuine batched inference
# ============================================================

def eval_worker(args, indexed_samples, eval_id):

    if not indexed_samples:
        return [], []

    device = f"cuda:{eval_id}"
    torch.cuda.set_device(eval_id)

    print(
        f"🚀 GPU {eval_id}: "
        f"{len(indexed_samples)} OCRBench samples, "
        f"batch_size={args.batch_size}"
    )

    model, processor = load_model(args, device)

    profiler = None
    warmup_done = False
    batch_latencies = []

    if args.measure_latency:
        profiler = Qwen3VLLatencyProfiler(model)
        profiler.install()

    # Keep original indices so we can restore benchmark order.
    dataset = OCRBenchDataset(
        indexed_samples,
        args.image_folder,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        collate_fn=OCRBenchCollator(processor),
    )

    generation_kwargs = get_generation_kwargs(args)

    # Store updated samples by their original indices.
    samples_by_index = {
        index: sample
        for index, sample in indexed_samples
    }

    try:
        for inputs, indices in tqdm(
            data_loader,
            desc=f"GPU {eval_id}",
        ):

            actual_batch_size = len(indices)

            if inputs["input_ids"].shape[0] != actual_batch_size:
                raise RuntimeError("Processor batch size mismatch.")

            if inputs["image_grid_thw"].shape[0] != actual_batch_size:
                raise RuntimeError(
                    "Expected exactly one image per sample."
                )

            inputs = inputs.to(device)

            prompt_length = inputs["input_ids"].shape[1]

            # ------------------------------------------------
            # Warmup (not measured)
            # ------------------------------------------------

            if args.measure_latency and not warmup_done:

                print(
                    f"🔥 GPU {eval_id}: "
                    f"{args.latency_warmup} warmup runs"
                )

                with torch.inference_mode():
                    for _ in range(args.latency_warmup):
                        model.generate(
                            **inputs,
                            **generation_kwargs,
                        )

                torch.cuda.synchronize(device)
                warmup_done = True

            # ------------------------------------------------
            # Timed or untimed generation
            # ------------------------------------------------

            if args.measure_latency:

                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

                profiler.start()

                try:
                    with torch.inference_mode():
                        generated_ids = model.generate(
                            **inputs,
                            **generation_kwargs,
                        )

                    latency = profiler.stop()
                    torch.cuda.synchronize(device)

                except Exception:
                    profiler.enabled = False
                    raise

                latency = dict(latency)

                prefill_ms = latency.get("prefill_ms")

                latency.update({
                    "gpu_id": eval_id,
                    "sample_indices": indices,
                    "batch_size": actual_batch_size,
                    "prompt_length_padded": prompt_length,
                    "generated_tokens_per_sample": int(
                        generated_ids.shape[1] - prompt_length
                    ),
                    "prefill_throughput_img_s": (
                        actual_batch_size * 1000.0 / prefill_ms
                        if prefill_ms is not None and prefill_ms > 0
                        else None
                    ),
                    "peak_memory_gib": (
                        torch.cuda.max_memory_allocated(device)
                        / (1024 ** 3)
                    ),
                })

                batch_latencies.append(latency)

            else:

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )

            # ------------------------------------------------
            # Decode every sample in the batch
            # ------------------------------------------------

            generated_ids_trimmed = generated_ids[
                :, prompt_length:
            ]

            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if len(outputs) != actual_batch_size:
                raise RuntimeError("Generated output count mismatch.")

            for index, output in zip(indices, outputs):
                sample = samples_by_index[index]

                # Preserve existing predictions when profiling
                # an already-evaluated checkpoint.
                if not (
                    args.measure_latency
                    and has_prediction(sample)
                ):
                    sample["predict"] = output.strip()

    finally:
        if profiler is not None:
            profiler.remove()

    return list(samples_by_index.items()), batch_latencies


# ============================================================
# Latency summary
# ============================================================

def summarize_latencies(args, batch_latencies):

    if not batch_latencies:
        return None

    def mean_metric(key):
        values = [
            record[key]
            for record in batch_latencies
            if record.get(key) is not None
        ]
        return float(statistics.mean(values)) if values else None

    total_samples = sum(
        record["batch_size"]
        for record in batch_latencies
    )

    total_prefill_ms = sum(
        record["prefill_ms"]
        for record in batch_latencies
        if record.get("prefill_ms") is not None
    )

    total_decode_ms = sum(
        record["decode_ms"]
        for record in batch_latencies
    )

    total_decode_steps = sum(
        record["decode_steps"]
        for record in batch_latencies
    )

    total_forward_ms = sum(
        record["total_forward_ms"]
        for record in batch_latencies
    )

    summary = {
        "model": args.save_name,
        "merge_strategy": args.merge_strategy,
        "compression_rate": args.compression_rate,
        "batch_size": args.batch_size,
        "num_gpus": args.num_workers,
        "num_samples": total_samples,
        "num_batches": len(batch_latencies),

        # All latencies are milliseconds per batch.
        "prefill_ms": mean_metric("prefill_ms"),
        "llm_prefill_ms": mean_metric("llm_prefill_ms"),
        "decode_total_ms": mean_metric("decode_ms"),
        "total_forward_ms": mean_metric("total_forward_ms"),

        # One decode step advances an entire batch.
        "decode_tpot_ms": (
            total_decode_ms / total_decode_steps
            if total_decode_steps > 0
            else None
        ),

        # Prefill-only throughput, images per second.
        "prefill_throughput_img_s": (
            total_samples * 1000.0 / total_prefill_ms
            if total_prefill_ms > 0
            else None
        ),

        # Model-forward-only throughput.
        "total_forward_throughput_img_s": (
            total_samples * 1000.0 / total_forward_ms
            if total_forward_ms > 0
            else None
        ),

        # Maximum allocated memory on any worker GPU.
        "peak_memory_gib": max(
            record["peak_memory_gib"]
            for record in batch_latencies
        ),
    }

    return summary


# ============================================================
# Main
# ============================================================

def main(args):

    if args.merge_strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported strategy: {args.merge_strategy}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.num_workers > torch.cuda.device_count():
        raise ValueError(
            f"Requested {args.num_workers} GPU workers, "
            f"but only {torch.cuda.device_count()} CUDA GPUs "
            f"are visible."
        )

    if args.merge_strategy in {"Fixed", "DRIP"}:
        if args.num_beams != 1:
            raise NotImplementedError(
                "Batched compressed inference currently "
                "supports num_beams=1."
            )

    os.makedirs(args.output_folder, exist_ok=True)

    output_path = os.path.join(
        args.output_folder,
        f"{args.save_name}.json",
    )

    # Resume from previous predictions if they exist.
    if os.path.exists(output_path):
        print(f"📂 Existing OCRBench results: {output_path}")
        with open(output_path, "r") as f:
            data = json.load(f)
    else:
        with open(args.OCRBench_file, "r") as f:
            data = json.load(f)

    # Profiling runs all samples, including those whose
    # predictions already exist. We don't overwrite those
    # predictions.
    #
    # Normal evaluation only runs missing predictions.
    indexed_samples = [
        (i, sample)
        for i, sample in enumerate(data)
        if args.measure_latency or not has_prediction(sample)
    ]

    print(f"📊 Total OCRBench records: {len(data)}")
    print(f"🚀 Records to process: {len(indexed_samples)}")
    print(f"🚀 Batch size: {args.batch_size}")
    print(f"🚀 GPU workers: {args.num_workers}")
    print(f"⏱️ Profiling: {args.measure_latency}")

    shards = split_list(
        indexed_samples,
        args.num_workers,
    )

    if args.num_workers == 1:
        worker_outputs = [
            eval_worker(args, shards[0], 0)
        ]
    else:
        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(processes=args.num_workers) as pool:
            worker_outputs = pool.starmap(
                eval_worker,
                [
                    (args, shard, gpu_id)
                    for gpu_id, shard in enumerate(shards)
                ],
            )

    batch_latencies = []

    # Restore original OCRBench order.
    for updated_samples, worker_latencies in worker_outputs:
        for index, sample in updated_samples:
            data[index] = sample

        batch_latencies.extend(worker_latencies)

    # Preserve original OCRBench scoring.
    for sample in data:
        score_sample(sample)

    save_json(data, output_path)
    print(f"💾 Saved predictions: {output_path}")

    # --------------------------------------------------------
    # Latency outputs
    # --------------------------------------------------------

    if args.measure_latency:

        # One record per B-sample batch, not per image.
        batch_latency_path = os.path.join(
            args.output_folder,
            f"{args.save_name}_batch_latency.json",
        )

        save_json(
            batch_latencies,
            batch_latency_path,
        )

        summary = summarize_latencies(
            args,
            batch_latencies,
        )

        if summary is not None:

            summary_path = os.path.join(
                args.output_folder,
                f"{args.save_name}_latency.json",
            )

            save_json(summary, summary_path)

            print("\n" + "=" * 60)
            print("⏱️ Qwen3-VL OCRBench Batched Latency Summary")
            print("=" * 60)

            for key, value in summary.items():
                print(f"{key}: {value}")

            print("=" * 60)


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_folder",
        type=str,
        default="./OCRBench_Images",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results",
    )

    parser.add_argument(
        "--OCRBench_file",
        type=str,
        default="./OCRBench/OCRBench.json",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
    )

    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="qwen3_vl",
    )

    parser.add_argument(
        "--conv_mode",
        type=str,
        default="qwen3_vl",
    )

    # Number of independent GPU processes.
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )

    # DataLoader workers per GPU process.
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
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
        "--measure-latency",
        action="store_true",
    )

    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=3,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    main(args)
