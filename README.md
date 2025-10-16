# Urdu → Roman Urdu Translation using Seq2Seq (BiLSTM + Attention)

This project translates **Urdu text** into **Roman Urdu** using a **sequence-to-sequence model with attention**.  
It uses **BPE tokenization** for Urdu and **WordPiece tokenization** for Roman Urdu.

---

## 🧠 Features
- Urdu text normalization and cleaning  
- Character Level Encoding for Urdu (source)  
- Character Level Encoding for Roman Urdu (target)  
- BiLSTM 2 layer Encoder, LSTM 4 layer Decoder 
- Mixed-precision training (AMP)  
- Training metrics: BLEU, CER, and Perplexity  
- Streamlit interface for live Romanization


