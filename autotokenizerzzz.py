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

# Load CSV
def load_user_data():
    if os.path.exists(DEFAULT_CSV):
        return pd.read_csv(DEFAULT_CSV)
    return pd.DataFrame(columns=["text", "label"])

def save_user_data(df):
    df.to_csv(DEFAULT_CSV, index=False)

user_data = load_user_data()

# Form Input
st.markdown("### ✍️ Tambah Contoh Data")
with st.form("add_example"):
    new_text = st.text_input("Masukkan teks:")
    label_input = st.selectbox("Label", LABELS)
    submit_btn = st.form_submit_button("Simpan ke CSV")

if submit_btn:
    if new_text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        new_entry = pd.DataFrame([{"text": new_text, "label": LABELS.index(label_input)}])
        if new_text in user_data["text"].values:
            st.warning("⚠️ Teks ini sudah ada dalam dataset.")
        else:
            combined = pd.concat([user_data, new_entry], ignore_index=True)
            combined = combined.drop_duplicates(subset=["text"], keep="first")
            save_user_data(combined)
            user_data = combined
            st.success("✅ Data disimpan.")

# Reset Dataset & Model
if st.button("🧹 Reset Dataset dan Model"):
    if os.path.exists(DEFAULT_CSV): os.remove(DEFAULT_CSV)
    if os.path.exists(DEFAULT_MODEL_PATH): os.remove(DEFAULT_MODEL_PATH)
    user_data = pd.DataFrame(columns=["text", "label"])
    st.success("✅ Dataset dan model berhasil direset.")

# Prediction
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

# Idiom-BERT Analysis
st.markdown("### 🧠 Analisis Idiom per Bahasa")
idioms = {
    "English": ["Break a leg"],
    "Indonesian": ["Buah bibir"],
    "Japanese": ["猫の手も借りたい"],
    "Thai": ["จับปลาสองมือ"],
    "Filipino": ["Itaga mo sa bato"]
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
            "BERT Known Since": "2019"
        })

if results:
    df_idioms = pd.DataFrame(results)
    st.markdown("### 🧾 Hasil Tabel Nama dan Idiom")
    st.dataframe(df_idioms)
    df_idioms.to_csv("idiom_analysis.csv", index=False)
