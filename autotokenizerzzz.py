import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

st.title("🧠 Analisis Idiom dan Bahasa dengan BERT + Generative Reason")

@st.cache_resource
def load_sbert():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

def generate_reason(prompt, tokenizer, model, max_length=60):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, temperature=0.7)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from output
    reason = text[len(prompt):].strip()
    return reason

sbert = load_sbert()
tokenizer_gpt2, model_gpt2 = load_gpt2()

with st.form("input_idiom"):
    language = st.text_input("Masukkan nama bahasa (language):", "English")
    idiom = st.text_input("Masukkan idiom:")
    submitted = st.form_submit_button("Analisis")

results = []

if submitted and idiom.strip() and language.strip():
    idiom_emb = sbert.encode(idiom, convert_to_tensor=True)
    context_text = f"Common idioms in {language}"
    lang_emb = sbert.encode(context_text, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(idiom_emb, lang_emb).item()
    valid = 1 if similarity > 0.3 else -1

    prompt = f"Explain why the idiom '{idiom}' is meaningful in the {language} language: "
    reason = generate_reason(prompt, tokenizer_gpt2, model_gpt2)

    results.append({
        "Language": language,
        "Idiom": idiom,
        "Similarity": similarity,
        "Validated": valid,
        "Reason": reason
    })

    df = pd.DataFrame(results)
    st.markdown("### Hasil Analisis Idiom dengan Reason Generatif")
    st.dataframe(df)
