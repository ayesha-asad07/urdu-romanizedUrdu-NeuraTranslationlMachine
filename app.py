import streamlit as st
import torch
from torch import nn
import json
import requests
import os

# --- Load model classes (import from your training code) ---
from model import EncoderBiLSTM, DecoderWithAttention, Seq2Seq, PAD_IDX, SOS_IDX, EOS_IDX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load vocabularies ---
with open("urdu_bpe.vocab", "r", encoding="utf-8") as f:
    urdu_vocab = [line.strip().split()[0] for line in f.readlines()]
with open("roman_wp-vocab.txt", "r", encoding="utf-8") as f:
    roman_vocab = [line.strip().split()[0] for line in f.readlines()]

urdu_stoi = {tok: i for i, tok in enumerate(urdu_vocab)}
roman_itos = {i: tok for i, tok in enumerate(roman_vocab)}

# --- Define inference helpers ---
def encode_urdu(sentence, stoi, sos_idx, eos_idx):
    tokens = sentence.strip().split()
    ids = [sos_idx] + [stoi.get(tok, stoi.get("<unk>", 3)) for tok in tokens] + [eos_idx]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def decode_roman(ids, itos):
    toks = []
    for i in ids:
        if i in [PAD_IDX, SOS_IDX, EOS_IDX]:
            continue
        toks.append(itos.get(int(i), "<unk>"))
    return " ".join(toks)

# --- Download model weights from Google Drive if not present ---
MODEL_PATH = "best_model_weights.pt"
DRIVE_ID = "1HJxyuPssBQWbRXXrV3JS1I-78NCYinqi"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("📦 Downloading model weights from Google Drive...")
    r = requests.get(DRIVE_URL)
    open(MODEL_PATH, "wb").write(r.content)
    st.success("✅ Model downloaded successfully!")

# --- Load trained model ---
INPUT_DIM = len(urdu_vocab)
OUTPUT_DIM = len(roman_vocab)

encoder = EncoderBiLSTM(INPUT_DIM, 256, 512, n_layers=2, dropout=0.3, pad_idx=PAD_IDX)
decoder = DecoderWithAttention(OUTPUT_DIM, 256, 512, n_layers=4, dropout=0.3, pad_idx=PAD_IDX)
model = Seq2Seq(encoder, decoder, device).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --- Streamlit UI ---
st.set_page_config(page_title="Urdu → Roman Urdu Transliterator", layout="centered")
st.title("🪶 Urdu → Roman Urdu Transliterator")

input_text = st.text_area("Enter Urdu text:", height=150, placeholder="مثال کے طور پر: محبت ایک خوبصورت جذبہ ہے")

if st.button("Transliterate"):
    with st.spinner("Converting..."):
        src = encode_urdu(input_text, urdu_stoi, SOS_IDX, EOS_IDX).to(device)
        with torch.no_grad():
            outputs = model(src, torch.tensor([src.size(1)]).to(device), trg=None, teacher_forcing=0.0, max_len=60)
        pred_ids = outputs.argmax(-1).squeeze(0).cpu().tolist()
        roman_text = decode_roman(pred_ids, roman_itos)

    st.subheader("Romanized Urdu Output:")
    st.success(roman_text)
