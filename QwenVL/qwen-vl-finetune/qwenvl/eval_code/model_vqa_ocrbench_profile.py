import json
from argparse import ArgumentParser
import torch
import os
from tqdm import tqdm
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Manager
import traceback
import sys, os
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor



FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../"))
sys.path.insert(0, PROJECT_ROOT)

from model.qwen3vl_compressed import CompressedQwen3VLForConditionalGeneration
from eval_code.latency_profiler import Qwen3VLLatencyProfiler


# ============================================================
# Utilities
# ============================================================

def split_list(lst, n):
    length = len(lst)
    avg = length // n

    result = []

    for i in range(n - 1):
        result.append(
            lst[i * avg:(i + 1) * avg]
        )

    result.append(
        lst[(n - 1) * avg:]
    )

    return result


def save_json(json_list, save_path):
    with open(save_path, "w") as file:
        json.dump(
            json_list,
            file,
            indent=4,
        )


# ============================================================
# Arguments
# ============================================================

def _get_args():
    parser = ArgumentParser()

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

    # If provided:
    #     model_path = LoRA checkpoint
    #     model_base = base Qwen model
    #
    # If None:
    #     model_path = full model
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

    # Kept for compatibility with existing evaluation scripts.
    parser.add_argument(
        "--conv_mode",
        type=str,
        default="qwen3_vl",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
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
        help="Measure prefill and decoding latency",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=3,
        help="Number of untimed warmup generations per GPU",
    )

    args = parser.parse_args()

    return args


# ============================================================
# OCRBench scores
# ============================================================

OCRBench_score = {
    "Regular Text Recognition": 0,
    "Irregular Text Recognition": 0,
    "Artistic Text Recognition": 0,
    "Handwriting Recognition": 0,
    "Digit String Recognition": 0,
    "Non-Semantic Text Recognition": 0,
    "Scene Text-centric VQA": 0,
    "Doc-oriented VQA": 0,
    "Key Information Extraction": 0,
    "Handwritten Mathematical Expression Recognition": 0,
}


AllDataset_score = {
    "IIIT5K": 0,
    "svt": 0,
    "IC13_857": 0,
    "IC15_1811": 0,
    "svtp": 0,
    "ct80": 0,
    "cocotext": 0,
    "ctw": 0,
    "totaltext": 0,
    "HOST": 0,
    "WOST": 0,
    "WordArt": 0,
    "IAM": 0,
    "ReCTS": 0,
    "ORAND": 0,
    "NonSemanticText": 0,
    "SemanticText": 0,
    "STVQA": 0,
    "textVQA": 0,
    "ocrVQA": 0,
    "ESTVQA": 0,
    "ESTVQA_cn": 0,
    "docVQA": 0,
    "infographicVQA": 0,
    "ChartQA": 0,
    "ChartQA_Human": 0,
    "FUNSD": 0,
    "SROIE": 0,
    "POIE": 0,
    "HME100k": 0,
}


num_all = {
    "IIIT5K": 0,
    "svt": 0,
    "IC13_857": 0,
    "IC15_1811": 0,
    "svtp": 0,
    "ct80": 0,
    "cocotext": 0,
    "ctw": 0,
    "totaltext": 0,
    "HOST": 0,
    "WOST": 0,
    "WordArt": 0,
    "IAM": 0,
    "ReCTS": 0,
    "ORAND": 0,
    "NonSemanticText": 0,
    "SemanticText": 0,
    "STVQA": 0,
    "textVQA": 0,
    "ocrVQA": 0,
    "ESTVQA": 0,
    "ESTVQA_cn": 0,
    "docVQA": 0,
    "infographicVQA": 0,
    "ChartQA": 0,
    "ChartQA_Human": 0,
    "FUNSD": 0,
    "SROIE": 0,
    "POIE": 0,
    "HME100k": 0,
}


# ============================================================
# Model loading
# ============================================================

def load_model(args, device):
    model_path = os.path.expanduser(
        args.model_path
    )

    if args.model_base is not None:
        # ----------------------------------------------------
        # LoRA checkpoint
        # ----------------------------------------------------

        model_base = os.path.expanduser(
            args.model_base
        )

        print(
            f"[{device}] Loading base model from "
            f"{model_base}"
        )


        if args.merge_strategy == "Fixed":
            print(f"🌊 Loading compressed Qwen3VL: Fixed, rate={args.compression_rate}")
            model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
                    model_base,
                    attn_implementation="flash_attention_2",
                    dtype=torch.bfloat16,
                    device_map={"": device},
            )
            model.model.set_compressor(merge_strategy=args.merge_strategy, compression_rate=args.compression_rate, temperature=args.sampling_temperature)

        elif args.merge_strategy == "DRIP":
            print(f"🌊 Loading compressed Qwen3VL: DRIP, rate={args.compression_rate}, temperature={args.sampling_temperature}")
            model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
                    model_base,
                    attn_implementation="flash_attention_2",
                    dtype=torch.bfloat16,
                    device_map="auto"
            )
            model.model.set_compressor(merge_strategy=args.merge_strategy, compression_rate=args.compression_rate, temperature=args.sampling_temperature, drip_path=args.drip_path, mlp_ratio=args.mlp_ratio)

        elif args.merge_strategy == "PruneSID":
            print(f"🌊 Loading compressed Qwen3VL: PruneSID, rate={args.compression_rate}")
            model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
                    model_base,
                    attn_implementation="flash_attention_2",
                    dtype=torch.bfloat16,
                    device_map="auto",
            )
            model.model.set_compressor(merge_strategy=args.merge_strategy, compression_rate=args.compression_rate, temperature=args.sampling_temperature)

        elif args.merge_strategy == "CDPruner":
            print(f"🌊 Loading compressed Qwen3VL: CDPruner, rate={args.compression_rate}")

            model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
                model_base,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
                device_map="auto",
            )

            model.model.set_compressor(merge_strategy="CDPruner", compression_rate=args.compression_rate)



        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_base,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )

        print(
            f"[{device}] Loading LoRA adapter from "
            f"{model_path}"
        )

        model = PeftModel.from_pretrained(
            model,
            model_path,
        )

        # Merge LoRA weights for inference.
        model = model.merge_and_unload()

        processor = AutoProcessor.from_pretrained(
            model_base,
        )

        model_name = os.path.basename(
            os.path.normpath(model_path)
        )

    else:
        # ----------------------------------------------------
        # Full checkpoint
        # ----------------------------------------------------

        print(
            f"[{device}] Loading model from "
            f"{model_path}"
        )

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )

        processor = AutoProcessor.from_pretrained(
            model_path,
        )

        model_name = os.path.basename(
            os.path.normpath(model_path)
        )

    model.eval()

    return model, processor, model_name


# ============================================================
# Evaluation worker
# ============================================================

def eval_worker(
    args,
    data,
    eval_id,
    output_queue,
):
    try:
        print(
            f"Process {eval_id} start."
        )

        device = f"cuda:{eval_id}"

        torch.cuda.set_device(eval_id)

        model, processor, model_name = load_model(
            args,
            device,
        )


        profiler = None
        warmup_done = False
        if args.measure_latency:
            profiler = Qwen3VLLatencyProfiler(model)
            profiler.install()
            print(f"⏱️ GPU {eval_id}: Latency profiling enabled")



        for i in tqdm(
            range(len(data)),
            desc=f"GPU {eval_id}",
        ):
            # ------------------------------------------------
            # Skip existing predictions
            # ------------------------------------------------

            if data[i].get("predict", 0) != 0:
                print(
                    f"{data[i]['image_path']} "
                    f"predict exist, continue."
                )

                continue

            # ------------------------------------------------
            # Image
            # ------------------------------------------------

            img_path = os.path.join(
                args.image_folder,
                data[i]["image_path"],
            )

            image = Image.open(
                img_path
            ).convert("RGB")

            # ------------------------------------------------
            # Question
            # ------------------------------------------------

            qs = data[i]["question"]

            qs = (
                qs
                + "\nAnswer the question using a "
                  "single word or phrase."
            )

            # ------------------------------------------------
            # Qwen multimodal message
            # ------------------------------------------------

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": qs,
                        },
                    ],
                }
            ]

            # Processor handles:
            #   - chat template
            #   - tokenization
            #   - image preprocessing
            #   - multimodal grid metadata
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = inputs.to(device)

            # ------------------------------------------------
            # Generation arguments
            # ------------------------------------------------

            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
                "use_cache": True,
            }

            if args.temperature > 0:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": args.temperature,
                    }
                )

                if args.top_p is not None:
                    generation_kwargs[
                        "top_p"
                    ] = args.top_p

            else:
                generation_kwargs[
                    "do_sample"
                ] = False

            # ------------------------------------------------
            # Generate
            # ------------------------------------------------

            if args.merge_strategy == "CDPruner":
                generation_kwargs["texts"] = [qs]


            # ------------------------------------------------
            # Latency measurement
            # ------------------------------------------------
            if args.measure_latency:
                # Warm up this worker's GPU/model once.
                # Reuse the current sample, but do not record timing.
                if not warmup_done:
                    print(
                        f"⏱️ GPU {eval_id}: "
                        f"Warming up ({args.latency_warmup} runs)"
                    )
                    with torch.inference_mode():
                        for _ in range(args.latency_warmup):
                            _ = model.generate(
                                **inputs,
                                **generation_kwargs,
                            )
                    torch.cuda.synchronize()
                    warmup_done = True
                # Start timing after preprocessing and warmup.
                torch.cuda.synchronize()
                profiler.start()

                try:
                    with torch.inference_mode():
                        generated_ids = model.generate(
                            **inputs,
                            **generation_kwargs,
                        )

                    latency = profiler.stop()

                except Exception:
                    profiler.enabled = False
                    raise

                # Save per-sample latency.
                data[i]["latency"] = latency

            else:
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )



            
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            # ------------------------------------------------
            # Remove prompt tokens
            # ------------------------------------------------

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            outputs = outputs.strip()

            data[i]["predict"] = outputs

        output_queue.put(
            {
                eval_id: data
            }
        )

        print(
            f"Process {eval_id} has completed."
        )

    except Exception:
        print(
            f"🔥 Worker {eval_id} crashed:"
        )

        traceback.print_exc()

        raise


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    multiprocessing.set_start_method(
        "spawn"
    )

    args = _get_args()

    os.makedirs(
        args.output_folder,
        exist_ok=True,
    )

    output_path = os.path.join(
        args.output_folder,
        f"{args.save_name}.json",
    )

    # --------------------------------------------------------
    # Resume from previous result
    # --------------------------------------------------------

    if os.path.exists(output_path):
        data_path = output_path

        print(
            f"output_path:{data_path} exist! "
            f"Only generate the results that were "
            f"not generated in {data_path}."
        )

    else:
        data_path = args.OCRBench_file

    with open(
        data_path,
        "r",
    ) as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Split benchmark across GPUs
    # --------------------------------------------------------

    data_list = split_list(
        data,
        args.num_workers,
    )

    output_queue = Manager().Queue()

    async_results = []

    pool = Pool(
        processes=args.num_workers
    )

    for i in range(
        len(data_list)
    ):
        result = pool.apply_async(
            eval_worker,
            args=(
                args,
                data_list[i],
                i,
                output_queue,
            ),
        )

        async_results.append(
            result
        )

    pool.close()
    pool.join()

    # Reveal hidden worker crashes.
    for result in async_results:
        result.get()

    # --------------------------------------------------------
    # Merge outputs from workers
    # --------------------------------------------------------

    results = {}

    while not output_queue.empty():
        result = output_queue.get()
        results.update(result)

    data = []

    for i in range(
        len(data_list)
    ):
        data.extend(
            results[i]
        )

    # ========================================================
    # OCRBench scoring
    # ========================================================

    for i in range(
        len(data)
    ):
        data_type = data[i]["type"]
        dataset_name = data[i]["dataset_name"]
        answers = data[i]["answers"]

        if data[i].get(
            "predict",
            0,
        ) == 0:
            continue

        predict = data[i]["predict"]

        data[i]["result"] = 0

        # ----------------------------------------------------
        # Mathematical expression recognition
        # ----------------------------------------------------

        if dataset_name == "HME100k":

            if isinstance(
                answers,
                list,
            ):
                for answer in answers:

                    answer = (
                        answer
                        .strip()
                        .replace("\n", " ")
                        .replace(" ", "")
                    )

                    normalized_predict = (
                        predict
                        .strip()
                        .replace("\n", " ")
                        .replace(" ", "")
                    )

                    if answer in normalized_predict:
                        data[i]["result"] = 1

            else:

                normalized_answer = (
                    answers
                    .strip()
                    .replace("\n", " ")
                    .replace(" ", "")
                )

                normalized_predict = (
                    predict
                    .strip()
                    .replace("\n", " ")
                    .replace(" ", "")
                )

                if (
                    normalized_answer
                    in normalized_predict
                ):
                    data[i]["result"] = 1

        # ----------------------------------------------------
        # Everything else
        # ----------------------------------------------------

        else:

            if isinstance(
                answers,
                list,
            ):
                for answer in answers:

                    answer = (
                        answer
                        .lower()
                        .strip()
                        .replace("\n", " ")
                    )

                    normalized_predict = (
                        predict
                        .lower()
                        .strip()
                        .replace("\n", " ")
                    )

                    if answer in normalized_predict:
                        data[i]["result"] = 1

            else:

                normalized_answer = (
                    answers
                    .lower()
                    .strip()
                    .replace("\n", " ")
                )

                normalized_predict = (
                    predict
                    .lower()
                    .strip()
                    .replace("\n", " ")
                )

                if (
                    normalized_answer
                    in normalized_predict
                ):
                    data[i]["result"] = 1

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_json(
        data,
        output_path,
    )

    # ========================================================
    # Latency summary
    # ========================================================

    if args.measure_latency:

        import numpy as np

        latencies = [
            sample["latency"]
            for sample in data
            if "latency" in sample
        ]

        if latencies:

            def mean_metric(key):
                values = [
                    x[key] for x in latencies
                    if x.get(key) is not None
                ]
                return float(np.mean(values)) if values else None

            total_decode_ms = sum(
                x["decode_ms"] for x in latencies
            )
            total_decode_steps = sum(
                x["decode_steps"] for x in latencies
            )

            summary = {
                "model": args.save_name,
                "merge_strategy": args.merge_strategy,
                "compression_rate": args.compression_rate,
                "num_samples": len(latencies),

                "prefill_ms": mean_metric("prefill_ms"),
                "llm_prefill_ms": mean_metric("llm_prefill_ms"),

                "decode_tpot_ms": (
                    total_decode_ms / total_decode_steps
                    if total_decode_steps > 0
                    else None
                ),

                "decode_total_ms": mean_metric("decode_ms"),
                "total_forward_ms": mean_metric("total_forward_ms"),
            }

            latency_path = os.path.join(
                args.output_folder,
                f"{args.save_name}_latency.json",
            )

            save_json(summary, latency_path)

            print("\n" + "=" * 55)
            print("⏱️ Qwen3-VL Latency Summary")
            print("=" * 55)

            for key, value in summary.items():
                print(f"{key}: {value}")

            print("=" * 55)
