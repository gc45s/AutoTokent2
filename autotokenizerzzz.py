import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
DEFAULT_CSV = "user_training_data.csv"
DEFAULT_MODEL_PATH = "offensive_model.pkl"

st.title("🛡️ Deteksi Konten Ofensif (Roberta Twitter)")

@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_roberta()

LABELS = ["not-offensive", "offensive"]

# Optional: Tambah data baru
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

# Reset CSV dan PKL
if st.button("🧹 Reset Dataset dan Model"):
    if os.path.exists(DEFAULT_CSV): os.remove(DEFAULT_CSV)
    if os.path.exists(DEFAULT_MODEL_PATH): os.remove(DEFAULT_MODEL_PATH)
    st.success("✅ Dataset dan model berhasil direset.")

# Prediksi
st.markdown("### 🔍 Cek Apakah Pesan Ofensif")
input_text = st.text_area("Masukkan teks untuk diperiksa:")

if st.button("Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        encoded = tokenizer(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = model(**encoded)
        probs = torch.nn.functional.softmax(output.logits, dim=-1).squeeze().numpy()
        pred = np.argmax(probs)

        if pred == 1:
            st.error(f"❌ Ofensif ({probs[pred]:.2f} confidence)")
        else:
            st.success(f"✅ Tidak ofensif ({probs[pred]:.2f} confidence)")

# Idiom Reasoning with Sentence-BERT
st.markdown("### 🧠 Analisis Idiom per Bahasa")
idioms = {
    "English": ["Break a leg"],
    "Japanese": ["猫の手も借りたい"]
}

sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

results = []
for lang, phrases in idioms.items():
    for idiom in phrases:
        lang_context = f"Common idioms in {lang}"
        idiom_emb = sbert.encode(idiom, convert_to_tensor=True)
        lang_emb = sbert.encode(lang_context, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(idiom_emb, lang_emb)
        valid = 1 if sim.item() > 0.3 else -1

        reason = f"'{idiom}' digunakan dalam konteks {lang.lower()} untuk menggambarkan situasi yang unik."
        name = f"{lang[:2]}-{idiom.split()[0].capitalize()}"

        results.append({
            "Language": lang,
            "Idiom": idiom,
            "Reason": reason,
            "Name": name,
            "Validated": valid,
            "BERT Known Since": "2019"  # Mock year
        })

if results:
    df_idioms = pd.DataFrame(results)
    st.markdown("### 🧾 Hasil Tabel Nama dan Idiom")
    st.dataframe(df_idioms)
    df_idioms.to_csv("idiom_analysis.csv", index=False)
