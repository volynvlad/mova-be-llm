#!/usr/bin/env python3
"""
Fine-tune Qwen model with Belarusian-enhanced tokenizer.

This script performs supervised fine-tuning of a Qwen model using the Belarusian
tokenizer to teach the model proper Belarusian language patterns.
"""

import logging
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from .config import DATA_PROCESSED_DIR
from .loader import load_directory, load_jsonl, load_text_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_belarusian_training_data(
    data_path: str | Path,
    max_samples: int = 10000,
    max_length: int = 512,
) -> Dataset:
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
            if len(text) > max_length * 4:
                text = text[: max_length * 4]
            texts.append(text)

        if len(texts) >= max_samples:
            break

    logger.info(f"Loaded {len(texts)} training samples")
    return Dataset.from_dict({"text": texts})


def prepare_dataset_for_training(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> Dataset:
    """Tokenize dataset for training."""

    def tokenize_function(examples):
        formatted_texts = []
        for text in examples["text"]:
            messages = [
                {"role": "user", "content": f"Працягні ці дапоўні тэкст: {text[:200]}..." if len(text) > 200 else text},
                {"role": "assistant", "content": text},
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(formatted)

        result = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )

        return result

    logger.info("Tokenizing dataset...")
    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    output_dir: str,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    max_length: int = 512,
) -> Trainer:
    """Create and configure Trainer."""

    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    total_steps = len(train_dataset) // effective_batch_size * num_train_epochs

    logger.info(f"Training setup:")
    logger.info(f"  Samples: {len(train_dataset)}")
    logger.info(f"  Batch size: {per_device_batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Epochs: {num_train_epochs}")
    logger.info(f"  Total steps: {total_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        # Training parameters
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Optimization
        optim="adamw_torch",
        warmup_steps=100,
        weight_decay=0.01,
        # Memory optimization
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        # Logging and saving
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        # Evaluation
        evaluation_strategy="no",
        # Other
        report_to=None,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    return trainer


def fine_tune_qwen_belarusian(args) -> None:
    """Main fine-tuning pipeline."""

    logger.info(f"Loading Belarusian tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer vocab size: {len(tokenizer.get_vocab()):,}")

    logger.info(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Resizing model embeddings...")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    logger.info(f"Model vocabulary resized to {len(tokenizer):,} tokens")

    logger.info(f"Loading training data from: {args.data_path}")
    raw_dataset = load_belarusian_training_data(args.data_path, args.max_samples, args.max_length)

    train_dataset = prepare_dataset_for_training(raw_dataset, tokenizer, args.max_length)

    trainer = create_trainer(
        model,
        tokenizer,
        train_dataset,
        args.output_dir,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    logger.info(f"Saving fine-tuned model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Fine-tuning completed!")

    logger.info("\n=== Testing fine-tuned model ===")
    test_prompts = [
        "Беларуская мова - гэта",
        "Мінск з'яўляецца",
        "Вясной у лесе",
    ]

    model.eval()
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {response}\n")


def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen model with Belarusian tokenizer")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model to fine-tune")
    parser.add_argument(
        "--tokenizer-path", default="./results/qwen2.5_7b_be_fixed_bytes", help="Path to Belarusian tokenizer"
    )
    parser.add_argument("--data-path", default="data/processed", help="Path to training data")
    parser.add_argument(
        "--output-dir", default="./results/qwen2.5_1.5b_be_finetuned", help="Output directory for fine-tuned model"
    )
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum training samples")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fine_tune_qwen_belarusian(args)


if __name__ == "__main__":
    main()
