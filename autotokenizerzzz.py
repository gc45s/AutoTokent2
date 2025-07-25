import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Language-Idiom Reason Analyzer", layout="wide")
st.title("🧠 Language & Idiom Reason Analyzer with BERT")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Input Section
st.markdown("### 🌐 Masukkan Bahasa dan Idiom")
lang = st.text_input("Bahasa:", placeholder="misalnya: Inggris")
idiom = st.text_input("Idiom:", placeholder="misalnya: Break the ice")
reason = st.text_area("Alasan (dalam Bahasa yang Sama):", placeholder="Mengapa idiom ini digunakan?")

if st.button("🔍 Analisis Idiom"):
    if not lang or not idiom or not reason:
        st.warning("Semua kolom wajib diisi.")
    else:
        name = f"{lang}_{idiom.replace(' ', '_')}"
        embeddings = model.encode([lang, idiom, reason])
        sim_lang = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_reason = util.cos_sim(embeddings[1], embeddings[2]).item()

        st.write(f"📛 **Nama Kombinasi**: `{name}`")
        st.write(f"🔁 *Kemiripan Bahasa ↔️ Idiom*: `{sim_lang:.2f}`")
        st.write(f"🧠 *Kemiripan Idiom ↔️ Alasan*: `{sim_reason:.2f}`")

        data = {
            "name": [name],
            "language": [lang],
            "idiom": [idiom],
            "reason": [reason],
            "similarity_lang_idiom": [sim_lang],
            "similarity_idiom_reason": [sim_reason]
        }
        df_result = pd.DataFrame(data)

        st.markdown("### 📊 Tabel Hasil")
        st.dataframe(df_result)

        df_result.to_csv("idiom_analysis.csv", index=False)
        st.success("📁 Data disimpan sebagai idiom_analysis.csv")
