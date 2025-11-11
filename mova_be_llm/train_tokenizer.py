import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data/interim/belarusian_cleaned.txt",
    model_prefix="bel_tokenizer",
    vocab_size=32000,
    model_type="bpe",
    character_coverage=1.0,
    control_symbols=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"],
    user_defined_symbols=["дз","Дз","дж","Дж","ў","Ў","і","І"]
)
