import os
from pathlib import Path

#-------- function for loading Dataset (walk through all poets)
def load_dataset(base_dir="dataset"):
    urdu_texts, roman_texts = [], []
    base_dir = Path(base_dir)

    for poet in base_dir.iterdir():
        ur_path = poet / "ur"
        en_path = poet / "en"

        if not ur_path.exists() or not en_path.exists():
            continue

        ur_files = sorted(os.listdir(ur_path))
        en_files = sorted(os.listdir(en_path))

        # only take aligned pairs
        for ur_file, en_file in zip(ur_files, en_files):
            ur_lines = open(ur_path / ur_file, encoding="utf-8").read().splitlines()
            en_lines = open(en_path / en_file, encoding="utf-8").read().splitlines()

            ur_lines = [l.strip() for l in ur_lines if l.strip()]
            en_lines = [l.strip().lower() for l in en_lines if l.strip()]

            min_len = min(len(ur_lines), len(en_lines))
            urdu_texts.extend(ur_lines[:min_len])
            roman_texts.extend(en_lines[:min_len])

    return urdu_texts, roman_texts


#-------- function for urdu corpus generation
def generate_urdu_corpus(texts):
    input_file = "corpus/urdu_corpus.txt"
    with open(input_file, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))



#-------- function for roman corpus generation
def generate_roman_corpus(texts):
    input_file = "corpus/roman_corpus.txt"
    with open(input_file, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))


#--------------- main
urdu_texts, roman_texts = load_dataset("dataset")
print("Samples:", urdu_texts[:2], roman_texts[:2])
print("Total pairs:", len(urdu_texts))

generate_urdu_corpus(urdu_texts)

generate_roman_corpus(roman_texts)
