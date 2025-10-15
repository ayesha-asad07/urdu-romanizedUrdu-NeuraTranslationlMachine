import csv
from typing import List, Tuple

SRC_FILE = "normalized_corpus/norm_ur.txt"
TGT_FILE = "normalized_corpus/norm_en.txt"

def create_csv_format() -> None:

    print("=" * 80)
    print("CREATING CSV FORMAT")
    print("=" * 80)

    # Read aligned files
    with open(SRC_FILE, "r", encoding="utf-8") as f:
        urdu_lines: List[str] = [line.strip() for line in f]
    with open(TGT_FILE, "r", encoding="utf-8") as f:
        english_lines: List[str] = [line.strip() for line in f]

    print(f"\nLoaded: {len(urdu_lines)} Urdu lines, {len(english_lines)} English lines")

    # Ensure alignment length matches; if not, truncate to smallest length.
    if len(urdu_lines) != len(english_lines):
        min_len = min(len(urdu_lines), len(english_lines))
        print(
            f"Warning: length mismatch detected. Truncating to {min_len} aligned pairs"
        )
        urdu_lines = urdu_lines[:min_len]
        english_lines = english_lines[:min_len]

    # Create pairs
    pairs: List[Tuple[str, str]] = list(zip(urdu_lines, english_lines))
    print(f"Created {len(pairs)} aligned pairs")

    # Write CSV
    print("\nCreating CSV format...")
    with open("roman_urdu.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["urdu", "roman"])
        for urdu, roman in pairs:
            writer.writerow([urdu, roman])
    print("Successfully created: corpus_roman_urdu.csv")

    # Verify CSV
    print("\nVerification:")
    with open("corpus_roman_urdu.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header if present
        csv_pairs = list(reader)
    print(f"CSV contains {len(csv_pairs)} pairs")

    if csv_pairs:
        print("\nFirst pair from CSV:")
        print(f"  Urdu:  {csv_pairs[0][0][:50]}...")
        print(f"  Roman: {csv_pairs[0][1][:50]}...")

    print("\n" + "=" * 80)
    print("CSV file created successfully")
    print("=" * 80)


if __name__ == "__main__":
    create_csv_format()