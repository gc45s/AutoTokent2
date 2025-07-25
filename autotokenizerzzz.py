import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
import joblib
from sentence_transformers import SentenceTransformer, util

# Konstanta
MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
DEFAULT_CSV = "user_training_data.csv"
DEFAULT_MODEL_PATH = "offensive_model.pkl"

LABELS = ["not-offensive", "offensive"]
IDIOM_LANGUAGES = {
    "English": ["Break a leg", "Piece of cake"],
    "Indonesian": ["Buah bibir", "Meja hijau"],
    "Japanese": ["猫の手も借りたい", "猿も木から落ちる"],
    "Thai": ["จับปลาสองมือ", "แมวไม่อยู่หนูร่าเริง"],
    "Filipino": ["Itaga mo sa bato", "Nagbibilang ng poste"]
}

# Caching model
@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_sbert():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

tokenizer, model = load_roberta()
sbert = load_sbert()

# --- Sidebar Navigation ---
st.sidebar.title("🛍️ Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Dashboard", "🛡️ Deteksi Teks", "🧠 Analisis Idiom", "🗂️ Manajemen Data"])

# --- Halaman Dashboard ---
if page == "🏠 Dashboard":
    st.title("📊 Dashboard Aplikasi Deteksi Ofensif dan Idiom")
    st.markdown("""
    Selamat datang di aplikasi analisis teks berbasis BERT. 
    Aplikasi ini memiliki fitur:
    - Deteksi konten ofensif dari teks
    - Analisis idiom khas dari berbagai bahasa
    - Manajemen dataset dan pelatihan ulang model

    > Powered by: `cardiffnlp/twitter-roberta-base-offensive` dan `Sentence-BERT`
    """)

# --- Halaman Deteksi ---
elif page == "🛡️ Deteksi Teks":
    st.title("🛡️ Deteksi Konten Ofensif")
    input_text = st.text_area("Masukkan teks:")
    if st.button("🔍 Deteksi"):
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

# --- Halaman Analisis Idiom ---
elif page == "🧠 Analisis Idiom":
    st.title("🧠 Analisis Idiom Berdasarkan Bahasa")
    results = []
    for lang, idiom_list in IDIOM_LANGUAGES.items():
        for idiom in idiom_list:
            lang_context = f"Common idioms in {lang}"
            idiom_emb = sbert.encode(idiom, convert_to_tensor=True)
            lang_emb = sbert.encode(lang_context, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(idiom_emb, lang_emb)
            valid = 1 if sim.item() > 0.3 else -1
            reason = f"'{idiom}' digunakan dalam konteks {lang.lower()}."
            name = f"{lang[:2]}-{idiom.split()[0].capitalize()}"

            results.append({
                "Language": lang,
                "Idiom": idiom,
                "Reason": reason,
                "Name": name,
                "Similarity Score": round(sim.item(), 2),
                "Valid": valid
            })

    df_idioms = pd.DataFrame(results)
    st.dataframe(df_idioms)
    df_idioms.to_csv("idiom_analysis.csv", index=False)
    st.success("📄 Analisis idiom selesai dan disimpan.")

# --- Halaman Manajemen Data ---
elif page == "🗂️ Manajemen Data":
    st.title("🗂️ Dataset dan Model")
    st.markdown("### ✍️ Tambah Contoh Teks")
    with st.form("form_data"):
        input_text = st.text_input("Teks:")
        input_label = st.selectbox("Label", LABELS)
        submit = st.form_submit_button("➕ Simpan")

    if submit:
        if input_text.strip() != "":
            label_val = LABELS.index(input_label)
            df_new = pd.DataFrame([{"text": input_text, "label": label_val}])
            if os.path.exists(DEFAULT_CSV):
                df_old = pd.read_csv(DEFAULT_CSV)
                df_combined = pd.concat([df_old, df_new]).drop_duplicates()
            else:
                df_combined = df_new
            df_combined.to_csv(DEFAULT_CSV, index=False)
            st.success("✅ Data ditambahkan.")
        else:
            st.warning("Teks tidak boleh kosong.")

    if st.button("🧹 Reset Dataset & Model"):
        if os.path.exists(DEFAULT_CSV): os.remove(DEFAULT_CSV)
        if os.path.exists(DEFAULT_MODEL_PATH): os.remove(DEFAULT_MODEL_PATH)
        st.success("🗑️ Dataset dan model dihapus.")
