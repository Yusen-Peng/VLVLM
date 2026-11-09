
import argparse
import base64
import io
import json
import math
import os
import sys

import pandas as pd
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../"))
sys.path.insert(0, PROJECT_ROOT)

# Reuse the existing Qwen3VL model-loading pipeline.
from eval_code.model_vqa_loader import load_model


all_options = ["A", "B", "C", "D"]


def split_list(lst, n):
    """Split a list into n roughly equal-sized chunks."""
    if n < 1:
        raise ValueError("num_chunks must be >= 1")

    chunk_size = math.ceil(len(lst) / n)

    if chunk_size == 0:
        return [lst.iloc[0:0]]

    return [
        lst.iloc[i:i + chunk_size]
        for i in range(0, len(lst), chunk_size)
    ]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)

    if k < 0 or k >= len(chunks):
        raise ValueError(
            f"Invalid chunk_idx={k}; available chunks: {len(chunks)}"
        )

    return chunks[k]


def is_none(value):
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in {"nan", "none", ""}

    result = pd.isna(value)
    return isinstance(result, (bool, type(pd.NA))) and bool(result) if result is not pd.NA else True


def get_options(row, options):
    parsed_options = []

    for option in options:
        value = row[option]

        if is_none(value):
            break

        parsed_options.append(str(value))

    return parsed_options


def load_image_from_base64(image_base64):
    image_bytes = base64.b64decode(image_base64)

    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def build_question(row, options, option_chars):
    question = str(row["question"])
    hint = row["hint"]

    if not is_none(hint):
        question = str(hint) + "\n" + question

    for option_char, option in zip(option_chars, options):
        question += "\n" + option_char + ". " + option

    return question


def eval_model(args):
    # ========================================================
    # Load Qwen3VL
    # ========================================================

    model, processor, model_name = load_model(args)

    # ========================================================
    # Load MMBench TSV
    # ========================================================

    questions = pd.read_table(
        os.path.expanduser(args.question_file)
    )

    questions = get_chunk(
        questions,
        args.num_chunks,
        args.chunk_idx,
    )

    answers_file = os.path.expanduser(args.answers_file)

    answer_dir = os.path.dirname(answers_file)

    if answer_dir:
        os.makedirs(answer_dir, exist_ok=True)

    print(f"Loaded {len(questions)} MMBench questions")

    # ========================================================
    # Inference
    # ========================================================

    with open(answers_file, "w") as ans_file:

        for _, row in tqdm(
            questions.iterrows(),
            total=len(questions),
        ):
            idx = row["index"]

            options = get_options(row, all_options)
            cur_option_char = all_options[:len(options)]

            num_rounds = (
                len(options)
                if args.all_rounds
                else 1
            )

            # Decode image once per question.
            image = load_image_from_base64(row["image"])

            for round_idx in range(num_rounds):

                # ============================================
                # Build MMBench Question
                # ============================================

                question = build_question(
                    row,
                    options,
                    cur_option_char,
                )

                # Preserve LLaVA's output prompt format.
                cur_prompt = question

                qs = question

                if args.single_pred_prompt:
                    if args.lang == "cn":
                        qs += "\n请直接回答选项字母。"
                    else:
                        qs += (
                            "\nAnswer with the option's "
                            "letter from the given choices directly."
                        )

                # ============================================
                # Qwen3VL Chat Template
                # ============================================

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

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                inputs = inputs.to(model.device)

                # ============================================
                # Generation Arguments
                # ============================================

                generation_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "num_beams": args.num_beams,
                    "use_cache": True,
                }

                if args.temperature > 0:
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": args.temperature,
                    })

                    if args.top_p is not None:
                        generation_kwargs["top_p"] = args.top_p

                else:
                    generation_kwargs["do_sample"] = False

                # CDPruner question-conditioned compression.
                if args.merge_strategy == "CDPruner":
                    generation_kwargs["texts"] = [cur_prompt]

                # ============================================
                # Qwen3VL Generation
                # ============================================

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )

                # Qwen3VL returns prompt + generated tokens.
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
                )[0].strip()

                # ============================================
                # Save LLaVA-compatible Output
                # ============================================

                ans_id = shortuuid.uuid()

                ans_file.write(
                    json.dumps(
                        {
                            "question_id": idx,
                            "round_id": round_idx,
                            "prompt": cur_prompt,
                            "text": outputs,
                            "options": options,
                            "option_char": cur_option_char,
                            "answer_id": ans_id,
                            "model_id": model_name,
                            "metadata": {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                ans_file.flush()

                # ============================================
                # Rotate options for the next round
                # ============================================

                options = options[1:] + options[:1]

                cur_option_char = (
                    cur_option_char[1:]
                    + cur_option_char[:1]
                )

    print(f"✅ MMBench inference completed: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ========================================================
    # Model Arguments
    # ========================================================

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
        "--question-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--answers-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--image-folder",
        type=str,
        default="",
    )

    parser.add_argument(
        "--conv-mode",
        type=str,
        default="qwen3_vl",
    )

    # ========================================================
    # Evaluation Arguments
    # ========================================================

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")

    # ========================================================
    # Compression Arguments
    # ========================================================

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

    args = parser.parse_args()

    eval_model(args)
