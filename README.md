# Urdu to Roman Urdu Transliteration using Seq2Seq (BiLSTM Encder, LSTM Decoder)

This project translates **Urdu text** into **Roman Urdu** using a **sequence-to-sequence model**.  
It uses **CLE tokenization** for Urdu and Roman Urdu.

---

## Features
- Urdu text normalization and cleaning  
- Character Level Encoding for Urdu (source)  
- Character Level Encoding for Roman Urdu (target)  
- BiLSTM 2 layer Encoder, LSTM 4 layer Decoder 
- Mixed-precision training (AMP)  
- Training metrics: BLEU, CER, and Perplexity  
- Streamlit interface for live Romanization


