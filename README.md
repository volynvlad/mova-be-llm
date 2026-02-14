# mova-be-llm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

LLM for belarusian language

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mova_be_llm and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mova_be_llm   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mova_be_llm a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Setup

```sh
sudo apt install libdjvulibre-dev pkg-config
```

```sh
uv sync
```

```sh
uv run pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

# How to get data?

## Scrape data

```sh

python mova_be_llm/scraper.py
# OR
uv run python -m mova_be_llm.scraper

```

## Extract Data

```sh

./bin/extract_books.sh

```

## Read / Parse book

```sh

# read
uv run python examples/read_book.py
# parse
uv run python mova_be_llm/parse_books.py

```

### Construct training text corpus

```sh

uv run python mova_be_llm/construct_corpus.py qwen data/interim/ data/processed/qwen_belarusian.txt
uv run python mova_be_llm/clean_txt.py

```

## Create training datasets

Generate instruction-response datasets from the book corpus in `data/interim/`:

### Step 1: Generate individual datasets

```sh
# Text continuation pairs (prefix → continuation)
uv run python -m mova_be_llm.create_continuation_dataset \
  --max-samples 50000 --seed 42

# Grammar correction, cloze fill-in-the-blank, spelling
uv run python -m mova_be_llm.create_grammar_dataset \
  --max-samples 30000 --error-rate 0.15 --seed 42

# QA: topic questions, factual extraction, true/false
uv run python -m mova_be_llm.create_qa_dataset \
  --max-samples 20000 --seed 42
```

### Step 2: Combine into final train/validation split

```sh
uv run python -m mova_be_llm.build_training_dataset \
  --input-files \
    data/processed/continuation_dataset.jsonl \
    data/processed/grammar_dataset.jsonl \
    data/processed/qa_dataset.jsonl \
  --val-split 0.05 --seed 42
```

This produces `data/processed/training_dataset.jsonl` and `data/processed/validation_dataset.jsonl`,
directly loadable by the fine-tuning pipeline.

# Train Tokenizer

## Train standalone tokenizer
```sh
uv run python mova_be_llm/train_belarusian_tokenizer.py \
    --corpus data/processed/qwen_belarusian_v2.txt \
    --output ./be_tokenizer \
    --vocab-size 32000
```

## Merge with Qwen2.5
```sh
uv run python mova_be_llm/train_belarusian_tokenizer.py \
    --dataset oscar \
    --merge-model Qwen/Qwen2.5-7B \
    --output ./qwen_be \
    --max-samples 100000
```

# Model Training and Usage

## Fine-tune Qwen with Belarusian tokenizer

### Method 1: Standard fine-tuning
Fine-tune a Qwen model using Belarusian-enhanced tokenizer:

```sh
# Quick test with small dataset
uv run python -m mova_be_llm.fine_tune_belarusian_qwen --max-samples 1000 --epochs 1

# Full training with custom parameters
uv run python -m mova_be_llm.fine_tune_belarusian_qwen \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --tokenizer-path ./results/qwen2.5_7b_be_fixed_bytes \
  --data-path data/processed \
  --output-dir ./results/qwen2.5_1.5b_be_finetuned \
  --max-samples 10000 \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 5e-5
```

### Method 2: Unsloth-optimized fine-tuning (Recommended)
Use Unsloth for 2x faster training and 80% memory reduction:

```sh
# Quick test with Unsloth
uv run python -m mova_be_llm.fine_tune_belarusian_qwen --max-samples 1000 --epochs 1

# Full training with Unsloth optimizations
uv run python -m mova_be_llm.fine_tune_belarusian_qwen \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --tokenizer-path ./results/qwen2.5_7b_be_fixed_bytes \
  --data-path data/processed \
  --output-dir ./results/qwen2.5_1.5b_be_unsloth \
  --max-samples 10000 \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4
```

**Unsloth Benefits:**
- 2x faster training speed
- 80% less memory usage
- 4-bit quantization support
- Optimized LoRA training
- Automatic gradient checkpointing

## Use Belarusian-enhanced tokenizer
Example showing how to use the merged tokenizer with Qwen models:

```sh
uv run python examples/use_belarusian_tokenizer.py
```

This script demonstrates:
- Tokenization comparison between original and Belarusian-enhanced tokenizers
- Text generation using the fine-tuned model
- Performance analysis showing 30-48% token reduction for Belarusian text

