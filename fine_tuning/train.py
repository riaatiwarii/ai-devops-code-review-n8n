"""
Fine-tune a code LLM for PR review using LoRA (PEFT).

Usage:
    python train.py \
        --model deepseek-ai/deepseek-coder-6.7b-instruct \
        --data_path ./dataset/reviews.jsonl \
        --output_dir ./lora_adapter \
        --epochs 3

Dataset format (each line in JSONL):
    {
        "diff": "--- old\n+++ new\n...",
        "language": "python",
        "review": { "issues": [...], "overall_score": 7.2, "summary": "..." }
    }
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert code reviewer. Analyze the code diff and return structured JSON review."""


def load_dataset(path: str) -> Dataset:
    """Load JSONL dataset and convert to prompt/completion pairs."""
    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if not item.get("diff") or not item.get("review"):
                continue

            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n"
                f"<|user|>\nLanguage: {item.get('language', 'unknown')}\n\n"
                f"=== DIFF ===\n{item['diff'][:4000]}\n=== END DIFF ===\n<|end|>\n"
                f"<|assistant|>\n"
            )
            completion = json.dumps(item["review"], indent=2)
            records.append({"prompt": prompt, "completion": completion})

    logger.info(f"Loaded {len(records)} training examples")
    return Dataset.from_list(records)


def tokenize(example: Dict, tokenizer, max_len: int = 2048) -> Dict:
    full_text = example["prompt"] + example["completion"] + tokenizer.eos_token
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    # Labels: mask the prompt tokens (-100 = ignore in loss)
    prompt_len = len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./lora_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.05)
    args = parser.parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(args.data_path)
    split = dataset.train_test_split(test_size=args.val_split, seed=42)

    def tokenize_fn(x):
        return tokenize(x, tokenizer, args.max_len)

    train_ds = split["train"].map(tokenize_fn, remove_columns=split["train"].column_names)
    val_ds = split["test"].map(tokenize_fn, remove_columns=split["test"].column_names)

    # ── Model (4-bit QLoRA) ───────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        logging_steps=10,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()