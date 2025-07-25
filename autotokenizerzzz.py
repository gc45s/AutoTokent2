import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Constants
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

# Load models with caching
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

# Sidebar Navigation
st.sidebar.title("🛍️ Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Dashboard", "🛡️ Text Detection", "🧠 Idiom Analysis", "🗂️ Data Management"])

# Dashboard Page
if page == "🏠 Dashboard":
    st.title("📊 Offensive Content and Idiom Analysis Dashboard")
    st.markdown("""
    Welcome to the BERT-based text analysis application.  
    Features include:
    - Offensive content detection  
    - Idiom analysis from various languages  
    - Dataset management and model retraining  
    \nPowered by `cardiffnlp/twitter-roberta-base-offensive` and `Sentence-BERT`
    """)

# Offensive Text Detection Page
elif page == "🛡️ Text Detection":
    st.title("🛡️ Offensive Content Detection")
    input_text = st.text_area("Enter text to analyze:")
    if st.button("🔍 Detect"):
        if not input_text.strip():
            st.warning("Text cannot be empty.")
        else:
            encoded = tokenizer(input_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                output = model(**encoded)
            probs = torch.nn.functional.softmax(output.logits, dim=-1).squeeze().numpy()
            pred = np.argmax(probs)

            if pred == 1:
                st.error(f"❌ Offensive ({probs[pred]:.2f} confidence)")
            else:
                st.success(f"✅ Not Offensive ({probs[pred]:.2f} confidence)")

# Idiom Analysis Page
elif page == "🧠 Idiom Analysis":
    st.title("🧠 Idiom Analysis Based on Input Data")
    st.info("Enter idioms and select language, then click **Analyze Idioms**. The 'Meaning' and 'Example Sentence' will be auto-translated.")

    # Idiom input table with dynamic rows
    idiom_input_df = st.data_editor(
        pd.DataFrame({
            "Idiom": ["Break a leg", "猫の手も借りたい"],
            "Language": ["English", "Japanese"]
        }),
        column_config={
            "Language": st.column_config.SelectboxColumn("Language", options=list(IDIOM_LANGUAGES.keys()))
        },
        num_rows="dynamic",
        use_container_width=True,
        key="idiom_input_editor"
    )

    if st.button("🔍 Analyze Idioms"):
        with st.spinner("Calculating similarity and translating..."):
            try:
                results = []
                for _, row in idiom_input_df.iterrows():
                    lang = row["Language"]
                    idiom = row["Idiom"]

                    if not lang or not idiom:
                        continue

                    # Language code for translation
                    lang_code = {
                        "English": "en",
                        "Indonesian": "id",
                        "Japanese": "ja",
                        "Thai": "th",
                        "Filipino": "tl"
                    }.get(lang, "en")

                    try:
                        meaning = GoogleTranslator(source='auto', target=lang_code).translate(idiom)
                    except Exception:
                        meaning = "(Translation failed)"

                    try:
                        example_sentence = GoogleTranslator(source='auto', target=lang_code).translate(f'Example usage of "{idiom}"')
                    except Exception:
                        example_sentence = "(Example translation failed)"

                    # Compute semantic similarity to language context
                    lang_context = f"Common idioms in {lang}"
                    idiom_emb = sbert.encode(idiom, convert_to_tensor=True)
                    lang_emb = sbert.encode(lang_context, convert_to_tensor=True)
                    sim = util.pytorch_cos_sim(idiom_emb, lang_emb)
                    valid = 1 if sim.item() > 0.3 else -1

                    # Articulate reason based on meaning and idiom
                    reason = f"'{idiom}' means: {meaning}."

                    # Generate simple unique name
                    name = f"{lang[:2]}-{idiom.split()[0].capitalize()}"

                    results.append({
                        "Language": lang,
                        "Idiom": idiom,
                        "Meaning": meaning,
                        "Example Sentence": example_sentence,
                        "Reason": reason,
                        "Name": name,
                        "Validated": valid,
                        "BERT Known Since": "2019"
                    })

                if results:
                    df_idiom_result = pd.DataFrame(results)
                    st.success("✅ Analysis complete.")
                    st.dataframe(df_idiom_result, use_container_width=True)
                    df_idiom_result.to_csv("idiom_analysis.csv", index=False)
                else:
                    st.warning("No valid idioms to analyze.")

            except Exception as e:
                st.error(f"Failed during analysis: {e}")

# Data Management Page
elif page == "🗂️ Data Management":
    st.title("🗂️ Dataset and Model Management")
    st.markdown("### ✍️ Add Example Text")
    with st.form("form_data"):
        input_text = st.text_input("Text:")
        input_label = st.selectbox("Label", LABELS)
        submit = st.form_submit_button("➕ Save")

    if submit:
        if input_text.strip():
            label_val = LABELS.index(input_label)
            df_new = pd.DataFrame([{"text": input_text, "label": label_val}])
            if os.path.exists(DEFAULT_CSV):
                df_old = pd.read_csv(DEFAULT_CSV)
                # Avoid duplicate entries
                df_combined = pd.concat([df_old, df_new]).drop_duplicates(ignore_index=True)
            else:
                df_combined = df_new
            df_combined.to_csv(DEFAULT_CSV, index=False)
            st.success("✅ Data added.")
        else:
            st.warning("Text cannot be empty.")

    if st.button("🧹 Reset Dataset & Model"):
        if os.path.exists(DEFAULT_CSV):
            os.remove(DEFAULT_CSV)
        if os.path.exists(DEFAULT_MODEL_PATH):
            os.remove(DEFAULT_MODEL_PATH)
        st.success("🗑️ Dataset and model cleared.")
