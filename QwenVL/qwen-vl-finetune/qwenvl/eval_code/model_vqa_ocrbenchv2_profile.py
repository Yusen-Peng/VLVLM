"""OCRBench-v2 evaluation with optional Qwen3-VL CUDA-event latency profiling.

Run from the Qwen3-VL project tree with eval_code/latency_profiler.py available.
The profiler measures model forwards during generate(), excluding preprocessing.
"""

import json
import os
import sys
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../"))
sys.path.insert(0, PROJECT_ROOT)

from model.qwen3vl_compressed import CompressedQwen3VLForConditionalGeneration
from eval_code.latency_profiler import Qwen3VLLatencyProfiler


def save_json(obj, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    os.replace(temporary_path, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="lmms-lab/OCRBench-v2")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default="./pred_folder")
    parser.add_argument("--save_name", type=str, default="qwen3vl_ocrbench_v2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="qwen3_vl")
    parser.add_argument("--num_workers", type=int, default=1)  # compatibility; single GPU
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-strategy", type=str, default="none")
    parser.add_argument("--compression-rate", type=float, default=1.0)
    parser.add_argument("--sampling-temperature", type=float, default=0.1)
    parser.add_argument("--drip-path", type=str, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=1.0)
    parser.add_argument(
        "--measure-latency", action="store_true",
        help="Record CUDA-event prefill and decoding latency per sample",
    )
    parser.add_argument(
        "--latency-warmup", type=int, default=3,
        help="Untimed warmup generations on the first measured sample",
    )
    return parser.parse_args()


def build_question(question):
    return question.strip() + "\nAnswer the question using a single word or phrase."


def normalize_row(row):
    """Remove PIL image before JSON serialization; preserve OCRBench-v2 metadata."""
    return {k: v for k, v in row.items() if k != "image"}


def load_model(args, device):
    model_path = os.path.expanduser(args.model_path)

    if args.model_base is not None:
        model_base = os.path.expanduser(args.model_base)
        print(f"Loading base model from {model_base}")

        if args.merge_strategy in ("Fixed", "DRIP", "PruneSID", "CDPruner"):
            print(f"🌊 Loading compressed Qwen3VL: {args.merge_strategy}, "
                  f"rate={args.compression_rate}")
            model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
                model_base,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
                device_map={"": device},
            )
            if args.merge_strategy == "DRIP":
                model.model.set_compressor(
                    merge_strategy="DRIP",
                    compression_rate=args.compression_rate,
                    temperature=args.sampling_temperature,
                    drip_path=args.drip_path,
                    mlp_ratio=args.mlp_ratio,
                )
            elif args.merge_strategy == "CDPruner":
                model.model.set_compressor(
                    merge_strategy="CDPruner",
                    compression_rate=args.compression_rate,
                )
            else:
                model.model.set_compressor(
                    merge_strategy=args.merge_strategy,
                    compression_rate=args.compression_rate,
                    temperature=args.sampling_temperature,
                )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_base,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )

        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(model_base)
        model_name = os.path.basename(os.path.normpath(model_path))
    else:
        if args.merge_strategy != "none":
            print("[warning] --merge-strategy applies only when --model_base is set "
                  "(matching the original OCRBench-v2 loader).")
        print(f"Loading model from {model_path}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
        processor = AutoProcessor.from_pretrained(model_path)
        model_name = os.path.basename(os.path.normpath(model_path))

    model.eval()
    return model, processor, model_name


def generate_answer(model, processor, image, question, args, device, profiler=None):
    qs = build_question(question)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": qs},
        ],
    }]

    # Image preprocessing and tokenization are NOT included in the latency.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "use_cache": True,
    }
    if args.temperature > 0:
        generation_kwargs.update(do_sample=True, temperature=args.temperature)
        if args.top_p is not None:
            generation_kwargs["top_p"] = args.top_p
    else:
        generation_kwargs["do_sample"] = False

    if args.merge_strategy == "CDPruner":
        generation_kwargs["texts"] = [qs]

    if profiler is not None:
        # Warm up with the first sample; do not include warmups in predictions or timing.
        if not getattr(profiler, "_ocrbenchv2_warmed_up", False):
            print(f"⏱️ Warming up ({args.latency_warmup} generations)")
            with torch.inference_mode():
                for _ in range(args.latency_warmup):
                    _ = model.generate(**inputs, **generation_kwargs)
            torch.cuda.synchronize()
            profiler._ocrbenchv2_warmed_up = True

        torch.cuda.synchronize()
        profiler.start()
        try:
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generation_kwargs)
            latency = profiler.stop()
        except Exception:
            profiler.enabled = False
            raise
    else:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        latency = None

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return answer, latency


def summarize_latencies(results, args):
    latencies = [r["latency"] for r in results if isinstance(r.get("latency"), dict)]
    if not latencies:
        return None

    def mean_metric(key):
        values = [float(x[key]) for x in latencies if x.get(key) is not None]
        return sum(values) / len(values) if values else None

    total_decode_ms = sum(float(x.get("decode_ms") or 0) for x in latencies)
    total_decode_steps = sum(int(x.get("decode_steps") or 0) for x in latencies)

    return {
        "model": args.save_name,
        "dataset": args.dataset_path,
        "dataset_split": args.dataset_split,
        "merge_strategy": args.merge_strategy,
        "compression_rate": args.compression_rate,
        "num_samples": len(latencies),
        "prefill_ms": mean_metric("prefill_ms"),
        "llm_prefill_ms": mean_metric("llm_prefill_ms"),
        "decode_tpot_ms": (total_decode_ms / total_decode_steps
                            if total_decode_steps > 0 else None),
        "decode_total_ms": mean_metric("decode_ms"),
        "total_forward_ms": mean_metric("total_forward_ms"),
    }


def main():
    args = _get_args()
    if args.num_workers != 1:
        raise ValueError("This OCRBench-v2 profiling script is single-GPU: use --num_workers 1")
    if args.latency_warmup < 0:
        raise ValueError("--latency-warmup must be nonnegative")

    os.makedirs(args.output_folder, exist_ok=True)
    output_json = os.path.join(args.output_folder, f"{args.save_name}.json")
    latency_json = os.path.join(args.output_folder, f"{args.save_name}_latency.json")

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    device = "cuda:0"
    torch.cuda.set_device(0)

    model, processor, model_name = load_model(args, device)
    dataset = load_dataset(
        args.dataset_path,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
    )

    if args.resume and os.path.exists(output_json):
        results = load_json(output_json)
        start_idx = len(results)
        if start_idx > len(dataset):
            raise ValueError("Saved predictions exceed the dataset length; check the dataset/split")
        print(f"[resume] found {start_idx} existing predictions")
    else:
        results = []
        start_idx = 0

    profiler = None
    if args.measure_latency:
        profiler = Qwen3VLLatencyProfiler(model)
        profiler.install()
        print("⏱️ Qwen3-VL latency profiling enabled")

    try:
        for idx in tqdm(range(start_idx, len(dataset)), desc="ocrbench-v2"):
            row = dataset[idx]
            image = row["image"].convert("RGB")
            try:
                answer, latency = generate_answer(
                    model=model,
                    processor=processor,
                    image=image,
                    question=row["question"],
                    args=args,
                    device=device,
                    profiler=profiler,
                )
            finally:
                image.close()

            result_row = normalize_row(row)
            result_row["predict"] = answer
            if latency is not None:
                result_row["latency"] = latency
            results.append(result_row)

            if len(results) % 50 == 0:
                save_json(results, output_json)

        save_json(results, output_json)
        print(f"[saved] {output_json}")

        if args.measure_latency:
            summary = summarize_latencies(results, args)
            if summary is not None:
                save_json(summary, latency_json)
                print("\n" + "=" * 55)
                print("⏱️ Qwen3-VL OCRBench-v2 Latency Summary")
                print("=" * 55)
                for key, value in summary.items():
                    print(f"{key}: {value}")
                print("=" * 55)
                print(f"[saved] {latency_json}")
            else:
                print("[warning] No latency records to summarize")
    finally:
        if profiler is not None:
            profiler.enabled = False
            # Profiler's hooks are attached to the model for this run only.
            if hasattr(profiler, "uninstall"):
                profiler.uninstall()


if __name__ == "__main__":
    main()
