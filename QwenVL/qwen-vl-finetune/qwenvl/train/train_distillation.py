import os
import logging
import pathlib
import sys
from pathlib import Path
from dataclasses import dataclass, field
import torch.nn.functional as F
import torch
import transformers

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from qwenvl.data.data_processor import make_supervised_data_module

from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments

from qwenvl.model.qwen3vl_compressed import CompressedQwen3VLForConditionalGeneration, CompressedTrainer


"""
    top-level Control for experiments here.
"""
MERGE_STRATEGY = "Fixed"
COMPRESSION_RATE = 0.25
TEMPERATURE = 0.01 # 0.01, 0.1, 1.0
MLP_RATIO = 4.0

# leverage the BP checkpoint from SigLIP2 experiment to transfer
DRIP_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp001_new_downsample_train_full/drip.bin" 

@dataclass
class DistillationArguments:
    teacher_model_path: str = field(default="Qwen/Qwen3-VL-4B-Instruct")
    student_compressed: bool = field(default=False)
    kd_weight: float = field(default=1.0)
    kd_temperature: float = field(default=1.0)


def build_student(model_args, training_args, kd_args):
    if kd_args.student_compressed:
        print("🌊🌊🌊 Initializing compressed Qwen3-VL student")
        model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16 if training_args.bf16 else None,
        )
        model.model.set_compressor(
            merge_strategy=MERGE_STRATEGY,
            compression_rate=COMPRESSION_RATE,
            temperature=TEMPERATURE,
            drip_path=DRIP_PATH or None,
            mlp_ratio=MLP_RATIO,
        )
    else:
        print("🌊🌊🌊 Initializing original Qwen3-VL student")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16 if training_args.bf16 else None,
        )
    return model


class DistillationTrainer(CompressedTrainer):
    def __init__(
        self,
        *args,
        teacher_model,
        kd_weight=1.0,
        kd_temperature=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature

        self._sft_loss_sum = 0.0
        self._kd_loss_sum = 0.0
        self._loss_count = 0


    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        student_outputs = model(**inputs)

        sft_loss = student_outputs.loss
        student_logits = student_outputs.logits

        self.teacher_model.eval()

        teacher_inputs = {k: v for k, v in inputs.items()if k != "labels"}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)

        teacher_logits = teacher_outputs.logits

        # Align teacher/student prediction positions
        keep_mask = getattr(student_outputs, "keep_mask", None)

        labels = inputs["labels"]

        if keep_mask is not None:
            # Current compressed implementation supports B=1.
            assert labels.shape[0] == 1
            keep = keep_mask[0].bool()
            # Map student positions back to original positions.
            original_positions = torch.where(keep)[0]
            # The student and teacher logits at these positions correspond to the same retained input tokens.
            teacher_logits = teacher_logits[:, original_positions, :]
            # Labels after compression.
            aligned_labels = labels[:, keep]

        else:
            aligned_labels = labels

        assert teacher_logits.shape == student_logits.shape, f"Teacher: {teacher_logits.shape}, Student: {student_logits.shape}"

        # Select supervised prediction positions
        student_logits = student_logits[:, :-1, :]
        teacher_logits = teacher_logits[:, :-1, :]
        valid_mask = aligned_labels[:, 1:] != -100

        # KL divergence
        T = self.kd_temperature

        student_logits = student_logits[valid_mask].float()
        teacher_logits = teacher_logits[valid_mask].float()

        if student_logits.shape[0] == 0:
            kd_loss = sft_loss.new_zeros(())
        else:
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)

            teacher_probs = F.softmax(teacher_logits / T, dim=-1)

            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)

        loss = sft_loss + self.kd_weight * kd_loss


        self._sft_loss_sum += sft_loss.detach().float().item()
        self._kd_loss_sum += kd_loss.detach().float().item()
        self._loss_count += 1



        boundary_loss = getattr(student_outputs, "boundary_loss", None)

        if boundary_loss is not None:
            self._last_boundary_loss = boundary_loss.detach().float().item()

        return (loss, student_outputs) if return_outputs else loss


    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._loss_count > 0:
            logs["sft_loss"] = self._sft_loss_sum / self._loss_count
            logs["kd_loss"] = self._kd_loss_sum / self._loss_count

            # Optional: log weighted KD contribution
            logs["kd_weighted_loss"] = self.kd_weight * logs["kd_loss"]

            # Reset accumulators after logging
            self._sft_loss_sum = 0.0
            self._kd_loss_sum = 0.0
            self._loss_count = 0
        return super().log(logs, *args, **kwargs)



def build_teacher(model_args, training_args, kd_args):
    print("🥶🥶🥶 Initializing Qwen3-VL teacher")

    from transformers.integrations.deepspeed import (
        is_deepspeed_zero3_enabled,
        unset_hf_deepspeed_config,
    )

    # Teacher must not be initialized with ZeRO-3 partitioning.
    if is_deepspeed_zero3_enabled():
        unset_hf_deepspeed_config()

    teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(
        kd_args.teacher_model_path,
        cache_dir=training_args.cache_dir,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    teacher_model.requires_grad_(False)
    teacher_model.eval()
    teacher_model.config.use_cache = False

    embedding = teacher_model.get_input_embeddings()

    print(
        f"🥶 Teacher embedding shape: {embedding.weight.shape}",
        flush=True,
    )

    assert embedding.weight.ndim == 2, (
        f"Teacher embedding is still partitioned: "
        f"{embedding.weight.shape}"
    )

    return teacher_model




def configure_student(model, model_args, training_args, kd_args):
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("🌊 LoRA enabled")
        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        if kd_args.student_compressed:
            if MERGE_STRATEGY == "DRIP":
                for name, p in model.named_parameters():
                    # disable BP training
                    if "compressor.boundary_predictor" in name:
                        p.requires_grad = False
                print("🌊🌊🌊 LoRA training - BP won't be further trained!")
    else:
        from qwenvl.train.train_qwen import set_model
        set_model(model_args, model)

    return model



def train():

    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            DistillationArguments,
        )
    )

    model_args, data_args, training_args, kd_args = (
        parser.parse_args_into_dataclasses()
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    data_args.model_type = "qwen3vl"

    # --------------------------------------------------
    # Student
    # --------------------------------------------------

    student = build_student(
        model_args,
        training_args,
        kd_args,
    )

    student.config.use_cache = False

    # --------------------------------------------------
    # Gradient checkpointing
    # --------------------------------------------------

    if training_args.gradient_checkpointing:

        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

        else:

            def make_inputs_require_grad(module, inputs, output):
                output.requires_grad_(True)

            student.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    # --------------------------------------------------
    # Processor and tokenizer
    # --------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()

    # --------------------------------------------------
    # LoRA
    # --------------------------------------------------

    student = configure_student(
        student,
        model_args,
        training_args,
        kd_args,
    )

    # --------------------------------------------------
    # Teacher
    # --------------------------------------------------

    teacher = build_teacher(model_args, training_args, kd_args)
    teacher.to(training_args.device)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    data_module = make_supervised_data_module(
        processor,
        data_args=data_args,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        kd_weight=kd_args.kd_weight,
        kd_temperature=kd_args.kd_temperature,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):

        logging.info("Checkpoint found, resuming training")

        trainer.train(resume_from_checkpoint=True)

    else:

        trainer.train()

    trainer.save_state()

    student.config.use_cache = True

    trainer.save_model(training_args.output_dir)

    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
