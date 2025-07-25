import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
import joblib
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
            df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates()
        else:
            df_combined = df_new
        df_combined.to_csv(DEFAULT_CSV, index=False)
        st.success("✅ Data disimpan.")

        # Analisis idiom otomatis dari input
        sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        idiom_emb = sbert.encode(text_input, convert_to_tensor=True)

        lang_idioms = {
            "English": ["Break a leg", "Hit the sack", "Let the cat out of the bag"],
            "Indonesian": ["Banting tulang", "Buah tangan", "Naik darah"],
            "Japanese": ["猫の手も借りたい", "猿も木から落ちる"],
            "Thai": ["น้ำขึ้นให้รีบตัก", "จับปลาสองมือ"],
            "Filipino": ["Itaga mo sa bato", "Nagbibilang ng poste"]
        }

        idiom_results = []

        for lang, phrases in lang_idioms.items():
            for phrase in phrases:
                phrase_emb = sbert.encode(phrase, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(idiom_emb, phrase_emb).item()

                if sim > 0.6:
                    idiom_results.append({
                        "Language": lang,
                        "Idiom": phrase,
                        "Similarity": f"{sim:.2f}",
                        "Matched With Input": text_input
                    })

        if idiom_results:
            df_matches = pd.DataFrame(idiom_results)
            st.markdown("### 🧠 Hasil Analisis Idiom dari Input")
            st.dataframe(df_matches)
        else:
            st.info("🔎 Tidak ditemukan idiom yang relevan dengan input.")

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
