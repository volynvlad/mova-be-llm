#!/usr/bin/env python3
"""
Fine-tune Qwen model with Belarusian-enhanced tokenizer using Unsloth.

This script uses Unsloth for efficient fine-tuning of Qwen models with Belarusian
tokenizer, providing significantly faster training and lower memory usage.
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json
import sys
from trl.trainer import sft_trainer
import argparse
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from loguru import logger

# from .config import PROCESSED_DATA_DIR
from .loader import load_directory, load_jsonl, load_text_file

logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", backtrace=True)


def load_belarusian_training_data(data_path: str | Path, max_samples: int, max_length: int) -> Dataset:
    """Load Belarusian text data for training."""
    data_path = Path(data_path)
    texts = []

    if data_path.is_dir():
        logger.info(f"Loading texts from directory: {data_path}")
        corpus = load_directory(str(data_path))
    elif data_path.suffix == ".jsonl":
        logger.info(f"Loading texts from JSONL: {data_path}")
        corpus = load_jsonl(str(data_path))
    else:
        logger.info(f"Loading texts from file: {data_path}")
        corpus = load_text_file(str(data_path))

    for text in corpus:
        if text.strip() and len(text) > 20:
            # Truncate very long texts
            if len(text) > max_length * 4:  # Rough estimate
                text = text[: max_length * 4]
            texts.append(text)

        if len(texts) >= max_samples:
            break

    logger.info(f"Loaded {len(texts)} training samples")
    return Dataset.from_dict({"text": texts})


def formatting_prompts_func(example):
    conversations_list = example["conversations"]

    if conversations_list and isinstance(conversations_list[0], dict):
        conversations_list = [conversations_list]

    all_texts = []
    for conversations in conversations_list:
        texts = []
        for conv in conversations:
            if conv["from"] == "human":
                texts.append(f"<|im_start|>user\n{conv['value']}<|im_end|>")
            elif conv["from"] == "gpt":
                texts.append(f"<|im_start|>assistant\n{conv['value']}<|im_end|>")
        all_texts.append("".join(texts))

    return all_texts


def load_conversations_jsonl(path: str | Path, max_samples: int) -> Dataset:
    """Load a JSONL file that already contains conversations format."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "conversations" in record:
                records.append(record)
            if len(records) >= max_samples:
                break
    logger.info(f"Loaded {len(records)} conversation records from {path}")
    return Dataset.from_list(records)


def prepare_dataset_for_unsloth(dataset: Dataset, max_samples: int = 1000) -> Dataset:
    conversations = []
    for text in dataset["text"][:max_samples]:
        if len(text) > 100:
            split_point = min(200, len(text) // 2)
            instruction = text[:split_point].strip()
            response = text[split_point:].strip()
        else:
            instruction = "Працягні тэкст"
            response = text.strip()

        conversations.append(
            {"conversations": [{"from": "human", "value": instruction}, {"from": "gpt", "value": response}]}
        )

    return Dataset.from_list(conversations)


def generate_response(model, tokenizer, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    gen_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False
    outputs = model.generate(**gen_kwargs)

    return tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)


def fine_tune_qwen(args) -> tuple[FastLanguageModel, AutoTokenizer]:
    logger.info("Loading model and tokenizer...")
    model, _ = FastLanguageModel.from_pretrained(args.base_model, args.max_length, None, load_in_4bit=True)

    logger.info(f"Loading Belarusian tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Resizing model embeddings for Belarusian tokenizer to {len(tokenizer)}...")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=137,
        use_rslora=False,
        loftq_config=None,
    )

    logger.info("Model prepared with LoRA adapters")

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    logger.info(f"Loading training data from: {args.data_path}")
    data_path = Path(args.data_path)
    if data_path.suffix == ".jsonl":
        # Check if it's a conversations-format JSONL
        with open(data_path, "r", encoding="utf-8") as f:
            first_line = json.loads(f.readline())
        if "conversations" in first_line:
            logger.info("Detected conversations JSONL format, loading directly")
            train_dataset = load_conversations_jsonl(data_path, args.max_samples)
        else:
            raw_dataset = load_belarusian_training_data(args.data_path, args.max_samples, args.max_length)
            train_dataset = prepare_dataset_for_unsloth(raw_dataset, args.max_samples)
    else:
        raw_dataset = load_belarusian_training_data(args.data_path, args.max_samples, args.max_length)
        train_dataset = prepare_dataset_for_unsloth(raw_dataset, args.max_samples)

    response_template = "\n<|im_start|>assistant\n"
    trainer = sft_trainer.SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch_4bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=137,
            output_dir=args.output_dir,
            report_to=None,
            save_steps=500,
            save_total_limit=2,
        ),
        formatting_func=formatting_prompts_func,
        data_collator=DataCollatorForLanguageModeling(
            # response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    logger.info("Starting fine-tuning...")
    trainer_stats = trainer.train()
    logger.info(f"Training completed. Final loss: {trainer_stats.training_loss:.4f}")

    logger.info(f"Saving model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("\n=== Testing fine-tuned model ===")
    FastLanguageModel.for_inference(model)  # Enable 2x faster inference
    return model, tokenizer


def run_test_prompts(model, tokenizer, args):
    test_prompts = [
        "Беларуская мова - гэта",
        "Мінск з'яўляецца",
        "Вясной у лесе",
    ]

    for prompt in test_prompts:
        response = generate_response(model, tokenizer, prompt)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {response}\n")

    logger.info("Fine-tuning completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("Note: This is a LoRA adapter. Use FastLanguageModel.merge_and_save() for full model.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model with Belarusian tokenizer")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--tokenizer-path", default="./results/qwen2.5_7b_be_fixed_bytes", help="Path to Belarusian tokenizer")
    parser.add_argument("--data-path", default="data/processed/training_dataset.jsonl", help="Path to training data (JSONL with conversations or directory)")
    parser.add_argument("--output-dir", default="./results/qwen2.5_1.5b_be", help="Output directory for fine-tuned model")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum training samples")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, tokenizer = fine_tune_qwen(args)
    run_test_prompts(model, tokenizer, args)


if __name__ == "__main__":
    main()
