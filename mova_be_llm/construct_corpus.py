import os

import typer

MODEL_TO_FILE_SEP_TOKEN = {"qwen": " <|file_sep|> "}


def main(modelname: str, input_path: str, result_corpus_file: str):
    if modelname not in MODEL_TO_FILE_SEP_TOKEN:
        raise ValueError(f"{modelname} not supported. Only {list(MODEL_TO_FILE_SEP_TOKEN)}")
    with open(result_corpus_file, "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk(input_path):
            for file in files:
                with open(os.path.join(root, file), "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                outfile.write(MODEL_TO_FILE_SEP_TOKEN[modelname])


if __name__ == "__main__":
    typer.run(main)
