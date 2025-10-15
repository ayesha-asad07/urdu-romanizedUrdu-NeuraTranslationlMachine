import re
import unicodedata

# Input files
EN_INPUT = "corpus/roman_corpus.txt"
UR_INPUT = "corpus/urdu_corpus.txt"

# Output files
EN_OUTPUT = "normalized_corpus/norm_en.txt"
UR_OUTPUT = "normalized_corpus/norm_ur.txt"

# ---------------- ENGLISH NORMALIZATION ---------------- #
def normalize_english(text: str) -> str:
    # Lowercase
    text = text.lower()

    # Remove accents/diacritics
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])

    # Remove punctuation (keep only letters, digits, and spaces)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- URDU NORMALIZATION ---------------- #
def normalize_urdu(text: str) -> str:
    # Replace different forms of Alef
    text = re.sub(r"[أإآٱا]", "ا", text)

    # Replace different forms of Yeh
    text = re.sub(r"[يۍېے]", "ی", text)

    # Replace different forms of Heh
    text = re.sub(r"[ھۀة]", "ہ", text)

    # Remove tatweel
    text = text.replace("ـ", "")

    # Remove diacritics (harakat)
    urdu_diacritics = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
    text = urdu_diacritics.sub("", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- PROCESS BOTH FILES ---------------- #
def main():
    with open(EN_INPUT, "r", encoding="utf-8") as en_in, \
         open(UR_INPUT, "r", encoding="utf-8") as ur_in, \
         open(EN_OUTPUT, "w", encoding="utf-8") as en_out, \
         open(UR_OUTPUT, "w", encoding="utf-8") as ur_out:

        for en_line, ur_line in zip(en_in, ur_in):
            en_norm = normalize_english(en_line)
            ur_norm = normalize_urdu(ur_line)

            # Skip empty lines (remove blank lines)
            if en_norm or ur_norm:
                en_out.write(en_norm + "\n")
                ur_out.write(ur_norm + "\n")

    print(f"Normalized corpora created: {EN_OUTPUT}, {UR_OUTPUT}")


if __name__ == "__main__":
    main()
