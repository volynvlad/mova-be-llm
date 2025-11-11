import sentencepiece as spm

TOKENIZER_PATH = "./models/tokenizer/bel_tokenizer.model"

sp = spm.SentencePieceProcessor(TOKENIZER_PATH)
print(sp.Encode("Прывіт, як справы?"))
