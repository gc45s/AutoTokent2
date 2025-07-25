import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd

st.title("🧠 Idiom Relevancy Analyzer with Sentence-BERT")

# Load model sekali saja (cache_resource)
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# Data idiom dan konteks bahasa
idioms = {
    "English": ["Break a leg", "Piece of cake", "Spill the beans"],
    "Japanese": ["猫の手も借りたい", "猿も木から落ちる", "一石二鳥"]
}

contexts = {
    "English": "Common English idioms used in everyday speech",
    "Japanese": "日本語でよく使われる慣用句"
}

threshold = 0.4  # similarity threshold untuk relevansi

results = []

for lang, idiom_list in idioms.items():
    context_text = contexts[lang]
    context_emb = model.encode(context_text, convert_to_tensor=True)
    for idiom in idiom_list:
        idiom_emb = model.encode(idiom, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(idiom_emb, context_emb).item()
        is_relevant = similarity > threshold
        reason = (
            f"Idiom '{idiom}' cocok dengan bahasa {lang} "
            f"dengan skor similarity {similarity:.2f}."
            if is_relevant
            else f"Idiom '{idiom}' tidak umum dalam bahasa {lang} "
            f"(skor similarity {similarity:.2f})."
        )
        results.append({
            "Language": lang,
            "Idiom": idiom,
            "Similarity": similarity,
            "Relevant": is_relevant,
            "Reason": reason
        })

df_results = pd.DataFrame(results)

st.markdown("### Hasil Analisis Idiom per Bahasa")
st.dataframe(df_results.style.applymap(
    lambda v: 'background-color: #d4edda' if v is True else ('background-color: #f8d7da' if v is False else ''),
    subset=['Relevant']
))

st.markdown("### Penjelasan Per Idiom")
for idx, row in df_results.iterrows():
    st.write(f"- **{row['Idiom']}** ({row['Language']}): {row['Reason']}")
