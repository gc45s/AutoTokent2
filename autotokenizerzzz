import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
import joblib

MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
DEFAULT_CSV = "user_training_data.csv"
DEFAULT_MODEL_PATH = "offensive_model.pkl"

st.set_page_config(page_title="Deteksi Konten Ofensif", page_icon="🛡️", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .block-container {
        padding: 2rem;
        border-radius: 1rem;
        background-color: white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Deteksi Konten Ofensif - Twitter Roberta")
st.caption("Model berdasarkan `cardiffnlp/twitter-roberta-base-offensive` dari Hugging Face")

@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_roberta()

LABELS = ["not-offensive", "offensive"]

# ==== Bagian: Tambah Data Training ====
st.markdown("### ✍️ Tambah Contoh Data ke Dataset")
with st.form("add_example"):
    col1, col2 = st.columns(2)
    with col1:
        text_input = st.text_input("Teks baru")
    with col2:
        label_input = st.selectbox("Label", LABELS)
    submit_btn = st.form_submit_button("➕ Simpan ke Dataset")

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
        st.success("✅ Data berhasil disimpan ke CSV.")
    else:
        st.warning("⚠️ Teks tidak boleh kosong.")

# ==== Tombol Reset ====
st.markdown("### 🧹 Reset Dataset & Model")
if st.button("Reset CSV dan Pickle"):
    if os.path.exists(DEFAULT_CSV): os.remove(DEFAULT_CSV)
    if os.path.exists(DEFAULT_MODEL_PATH): os.remove(DEFAULT_MODEL_PATH)
    st.success("✅ Dataset dan model berhasil direset.")

# ==== Deteksi Konten ====
st.markdown("### 🔍 Deteksi Konten Ofensif")
input_text = st.text_area("Masukkan teks yang ingin diperiksa:", height=150)

if st.button("🚨 Deteksi Sekarang"):
    if input_text.strip() == "":
        st.warning("⚠️ Masukkan teks terlebih dahulu.")
    else:
        encoded = tokenizer(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = model(**encoded)
        probs = torch.nn.functional.softmax(output.logits, dim=-1).squeeze().numpy()
        pred = np.argmax(probs)

        st.markdown("### Hasil Deteksi")
        if pred == 1:
            st.error(f"❌ Ofensif ({probs[pred]:.2f} confidence)")
        else:
            st.success(f"✅ Tidak Ofensif ({probs[pred]:.2f} confidence)")
