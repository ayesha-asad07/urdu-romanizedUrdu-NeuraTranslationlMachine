import streamlit as st
import torch
from torch import nn
import json
from train import EncoderBiLSTM, Decoder, Seq2Seq, PAD_IDX, SOS_IDX, EOS_IDX
import os
import gdown

# Google Drive file ID for your model
DRIVE_FILE_ID = "19xXRPYqg8tD6dgPRZAsG6X5PEyiUH92a"   # <-- replace with your actual ID
MODEL_PATH = "best_checkpoint.pt"

# Check if model exists locally, if not, download it
if not os.path.exists(MODEL_PATH):
    print("Downloading best_checkpoint.pt from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load vocabularies (JSONs) to convert coming input into tokens ---
with open("source_vocab.json", "r", encoding="utf-8") as f:
    src_vocab = json.load(f)["char_to_idx"]
with open("target_vocab.json", "r", encoding="utf-8") as f:
    trg_vocab = json.load(f)["char_to_idx"]

# Build inverse maps
src_stoi = src_vocab
trg_itos = {v: k for k, v in trg_vocab.items()}

# --- Encode/Decode helpers ---
def encode_urdu(text, vocab, sos_idx, eos_idx):
    ids = [sos_idx] + [vocab.get(ch, 3) for ch in text] + [eos_idx]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def decode_roman(ids, itos):
    toks = []
    for i in ids:
        if i in [PAD_IDX, SOS_IDX, EOS_IDX]:
            continue
        toks.append(itos.get(int(i), "<UNK>"))
    return "".join(toks).strip()

# --- Load trained model ---
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)

encoder = EncoderBiLSTM(INPUT_DIM, 256, 512, n_layers=2, dropout=0.3, pad_idx=PAD_IDX)
decoder = Decoder(OUTPUT_DIM, 256, 512, n_layers=4, dropout=0.3, pad_idx=PAD_IDX)
model = Seq2Seq(encoder, decoder, device).to(device)


checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- Streamlit UI ---
st.set_page_config(page_title="Urdu → Roman Urdu Transliterator", layout="centered")
st.title("🪶 Urdu → Roman Urdu Transliterator")

input_text = st.text_area("Enter Urdu text:", height=150, placeholder="مثال کے طور پر: محبت ایک خوبصورت جذبہ ہے")

if st.button("Transliterate"):
    with st.spinner("Converting..."):
        src = encode_urdu(input_text, src_stoi, SOS_IDX, EOS_IDX).to(device)
        src_len = torch.tensor([src.size(1)], dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(src, src_len, trg=None, teacher_forcing=0.0, max_len=100)
        pred_ids = outputs.argmax(-1).squeeze(0).cpu().tolist()
        roman_text = decode_roman(pred_ids, trg_itos)

    st.subheader("Romanized Urdu Output:")
    st.success(roman_text)






