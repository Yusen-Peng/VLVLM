import argparse
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import shortuuid
import os
import sys
from LLaVA_wrapper.llava_local.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA_wrapper.llava_local.conversation import conv_templates, SeparatorStyle
from LLaVA_wrapper.llava_local.model.builder import load_pretrained_model
from LLaVA_wrapper.llava_local.utils import disable_torch_init
from LLaVA_wrapper.llava_local.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from LLaVA_wrapper.llava_local.llava_latency_profiler import LLaVALatencyProfiler
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    model.eval()
    profiler = None
    warmup_done = False
    latencies = []
    profiler = LLaVALatencyProfiler(model)
    profiler.install()



    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    latency_file = answers_file + ".latency.jsonl"
    lat_file = open(latency_file, "w")


    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)


        # Prepare generation arguments
        generation_kwargs = {
            "images": image_tensor.to(
                dtype=torch.float16,
                device="cuda",
                non_blocking=True,
            ),
            "image_sizes": image_sizes,
            "do_sample": args.temperature > 0,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        }

        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature

            if args.top_p is not None:
                generation_kwargs["top_p"] = args.top_p

        # Preserve CDPruner / DRIP++ special handling.
        if model.get_vision_tower().merge_strategy in {"CDPruner", "DRIP++"}:
            question = cur_prompt.replace("\nAnswer the question using a single word or phrase.", "")
            generation_kwargs["texts"] = question


        # Warm up with the first actual sample.
        if not warmup_done:
            print(f"🔥 LLaVA warmup: 3 runs")

            with torch.inference_mode():
                for _ in range(3):
                    model.generate(
                        input_ids,
                        **generation_kwargs,
                    )

            torch.cuda.synchronize()
            warmup_done = True

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        profiler.start()

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    **generation_kwargs,
                )

            latency = profiler.stop()
            torch.cuda.synchronize()

        except Exception:
            profiler.enabled = False
            raise

        latency["question_id"] = idx
        latency["generated_tokens"] = int(output_ids.shape[1])
        latency["peak_memory_gib"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

        latencies.append(latency)

        lat_file.write(json.dumps(latency) + "\n")
        lat_file.flush()

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
    ans_file.close()

    lat_file.close()
    profiler.remove()

    def avg(key):
        import statistics
        values = [
            x[key]
            for x in latencies
            if x.get(key) is not None
        ]
        return (
            float(statistics.mean(values))
            if values
            else None
        )

    total_decode_ms = sum(x["decode_ms"] for x in latencies)

    total_decode_steps = sum(x["decode_steps"] for x in latencies)

    summary = {
        "model": model_name,
        "num_samples": len(latencies),
        "num_chunks": args.num_chunks,
        "chunk_idx": args.chunk_idx,

        "prefill_ms": avg("prefill_ms"),
        "llm_prefill_ms": avg("llm_prefill_ms"),

        "decode_tpot_ms": (
            total_decode_ms / total_decode_steps
            if total_decode_steps > 0
            else None
        ),

        "decode_total_ms": avg("decode_ms"),
        "total_forward_ms": avg("total_forward_ms"),

        "mean_generated_tokens": avg("generated_tokens"),
        "mean_decode_steps": avg("decode_steps"),

        "peak_memory_gib": max(
            x["peak_memory_gib"]
            for x in latencies
        ),
    }

    summary_path = (
        answers_file + ".latency_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("⏱️ LLaVA Latency Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
