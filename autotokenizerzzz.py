import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
import joblib
import re
import string

MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
DEFAULT_CSV = "user_training_data.csv"
DEFAULT_MODEL_PATH = "offensive_model.pkl"
LABELS = ["not-offensive", "offensive"]

st.title("🛡️ Deteksi Konten Ofensif (Twitter Roberta)")

# --- Preprocessing sesuai CardiffNLP ---
def preprocess(text):
    text = re.sub(r"@\w+", "", text)                     # Remove mentions
    text = re.sub(r"http\S+", "", text)                  # Remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text.strip()

# --- Load Model dan Tokenizer ---
@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, normalization=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_roberta()

# --- Tambah data pelatihan manual ---
st.markdown("### ✍️ Tambah Contoh Data")
with st.form("add_example"):
    text_input = st.text_input("Masukkan teks:")
    label_input = st.selectbox("Label", LABELS)
    submit_btn = st.form_submit_button("Simpan ke CSV")

if submit_btn:
    if text_input.strip() != "":
        label_val = LABELS.index(label_input)
        df_new = pd.DataFrame([{"text": text_input, "label": label_val}])
        if os.path.exists(DEFAULT_CSV):
            df_old = pd.read_csv(DEFAULT_CSV)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(DEFAULT_CSV, index=False)
        st.success("✅ Data disimpan.")
    else:
        st.warning("Teks tidak boleh kosong.")

# --- Reset Dataset dan Model ---
if st.button("🧹 Reset Dataset dan Model"):
    if os.path.exists(DEFAULT_CSV): os.remove(DEFAULT_CSV)
    if os.path.exists(DEFAULT_MODEL_PATH): os.remove(DEFAULT_MODEL_PATH)
    st.success("✅ Dataset dan model berhasil direset.")

# --- Deteksi Ofensif ---
st.markdown("### 🔍 Cek Apakah Pesan Ofensif")
input_text = st.text_area("Masukkan teks untuk diperiksa:")

if st.button("Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        clean_text = preprocess(input_text)
        encoded = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**encoded)
        probs = torch.nn.functional.softmax(output.logits, dim=-1).squeeze().numpy()
        pred = np.argmax(probs)

        if pred == 1:
            st.error(f"❌ Ofensif ({probs[pred]:.2f} confidence)")
        else:
            st.success(f"✅ Tidak ofensif ({probs[pred]:.2f} confidence)")
