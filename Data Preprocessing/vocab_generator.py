import json
from collections import Counter

SRC_FILE = "normalized_corpus/norm_ur.txt"
TGT_FILE = "normalized_corpus/norm_en.txt"

def generate_char_vocab(file_path, vocab_name):

    print(f"\nProcessing {vocab_name}...")
    
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found!")
        return None
    
    # Count character frequencies
    char_counter = Counter(text)
    
    # Remove newline from counting if you want
    # char_counter.pop('\n', None)
    
    # Sort characters by frequency (descending)
    sorted_chars = sorted(char_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocabulary mappings
    # Special tokens
    vocab = {
        '<PAD>': 0,  # Padding token
        '<SOS>': 1,  # Start of sequence
        '<EOS>': 2,  # End of sequence
        '<UNK>': 3   # Unknown token
    }
    
    # Add characters to vocabulary
    char_to_idx = vocab.copy()
    idx = len(vocab)
    
    for char, freq in sorted_chars:
        if char not in char_to_idx:
            char_to_idx[char] = idx
            idx += 1
    
    # Create reverse mapping (index to character)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    # Print statistics
    print(f"Total unique characters: {len(char_to_idx) - 4}")  # Excluding special tokens
    print(f"Vocabulary size (with special tokens): {len(char_to_idx)}")
    print(f"Total characters in text: {len(text)}")
    print(f"\nTop 10 most frequent characters:")
    for char, freq in sorted_chars[:10]:
        char_display = repr(char) if char in ['\n', '\t', ' '] else char
        print(f"  {char_display}: {freq}")
    
    return {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'char_frequencies': dict(sorted_chars),
        'vocab_size': len(char_to_idx),
        'total_chars': len(text)
    }

def save_vocab(vocab_dict, output_file):
    """Save vocabulary to a JSON file."""
    # Convert integer keys to strings for JSON serialization
    vocab_to_save = {
        'char_to_idx': vocab_dict['char_to_idx'],
        'idx_to_char': {str(k): v for k, v in vocab_dict['idx_to_char'].items()},
        'char_frequencies': vocab_dict['char_frequencies'],
        'vocab_size': vocab_dict['vocab_size'],
        'total_chars': vocab_dict['total_chars']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary saved to {output_file}")

def main():
    # Generate vocabulary for source (Urdu)
    source_vocab = generate_char_vocab(SRC_FILE, 'Urdu (Source)')
    
    if source_vocab:
        save_vocab(source_vocab, 'vocabulary/source_vocab.json')
    
    # Generate vocabulary for target (Roman)
    target_vocab = generate_char_vocab(TGT_FILE, 'Roman (Target)')

    if target_vocab:
        save_vocab(target_vocab, 'vocabulary/target_vocab.json')
    
    # Create combined vocabulary statistics
    if source_vocab and target_vocab:
        print("\n" + "="*60)
        print("VOCABULARY COMPARISON")
        print("="*60)
        print(f"Source (Urdu) vocabulary size: {source_vocab['vocab_size']}")
        print(f"Target (Roman) vocabulary size: {target_vocab['vocab_size']}")
        print(f"Source total characters: {source_vocab['total_chars']}")
        print(f"Target total characters: {target_vocab['total_chars']}")
        
        # Save combined statistics
        combined_stats = {
            'source': {
                'vocab_size': source_vocab['vocab_size'],
                'total_chars': source_vocab['total_chars'],
                'unique_chars': source_vocab['vocab_size'] - 4
            },
            'target': {
                'vocab_size': target_vocab['vocab_size'],
                'total_chars': target_vocab['total_chars'],
                'unique_chars': target_vocab['vocab_size'] - 4
            }
        }
        
        with open('vocabulary/vocab_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, ensure_ascii=False, indent=2)
        
        print("\nCombined statistics saved to vocab_statistics.json")

if __name__ == "__main__":
    main()