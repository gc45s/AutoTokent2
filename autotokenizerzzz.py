import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os
import joblib

# Config
LANGUAGE_MODELS = {
    "English": "cardiffnlp/twitter-roberta-base-offensive",
    "Indonesian": "indolem/indobert-base-uncased",
    "Japanese": "cl-tohoku/bert-base-japanese",
    "Thai": "airesearch/wangchanberta-base-att-spm-uncased",
    "Filipino": "bert-base-multilingual-cased"
}

# Use Sentence-BERT multilingual model for training embeddings
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

st.title("🛡️ Multilingual Offensive Content Detector & Trainer")

language = st.selectbox("Select Language", list(LANGUAGE_MODELS.keys()))

# Paths for user data and models per language
DATA_PATH = f"user_data_{language}.csv"
MODEL_PATH = f"clf_model_{language}.pkl"

@st.cache_resource(show_spinner=False)
def load_hf_model(lang):
    model_name = LANGUAGE_MODELS[lang]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_sbert():
    return SentenceTransformer(SBERT_MODEL_NAME)

def load_user_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame(columns=["text", "label"])

def save_user_data(df):
    df.to_csv(DATA_PATH, index=False)

def train_classifier(df, sbert_model):
    if df.empty or len(df) < 2:
        return None
    embeddings = sbert_model.encode(df["text"].tolist())
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, df["label"].values)
    joblib.dump(clf, MODEL_PATH)
    return clf

def load_classifier():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

# Load models
hf_tokenizer, hf_model = load_hf_model(language)
sbert_model = load_sbert()

# User data and classifier
user_data = load_user_data()
clf = load_classifier()
if clf is None:
    clf = train_classifier(user_data, sbert_model)

st.markdown("### 🔍 Offensive content detection using pretrained model")
input_text = st.text_area(f"Enter {language} text for detection:")

if st.button("Detect Offensive Content with HF Model"):
    if not input_text.strip():
        st.warning("Please enter some text!")
    else:
        inputs = hf_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = hf_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
        pred = probs.index(max(probs))
        if pred == 1:
            st.error(f"❌ Offensive content detected (confidence: {probs[pred]:.2f})")
        else:
            st.success(f"✅ Text is not offensive (confidence: {probs[pred]:.2f})")

st.markdown("---")
st.markdown("### 📝 Add new training example (trainable classifier)")

with st.form("add_example_form"):
    new_text = st.text_input(f"Enter new {language} text example:")
    new_label = st.selectbox("Label", ["Not Offensive", "Offensive"])
    submit = st.form_submit_button("Add Example")

if submit:
    if new_text.strip() == "":
        st.warning("Text cannot be empty.")
    else:
        label_num = 1 if new_label == "Offensive" else 0
        new_entry = pd.DataFrame([{"text": new_text, "label": label_num}])
        user_data = pd.concat([user_data, new_entry], ignore_index=True)
        save_user_data(user_data)
        st.success("✅ Example added. Retraining classifier...")
        clf = train_classifier(user_data, sbert_model)
        if clf:
            st.success("✅ Classifier retrained successfully.")
        else:
            st.warning("⚠️ Not enough data to train classifier yet.")

st.markdown("---")
if st.button("🧹 Reset user dataset and classifier"):
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    user_data = pd.DataFrame(columns=["text", "label"])
    clf = None
    st.success("✅ Dataset and classifier reset.")

st.markdown("---")
st.markdown("### 🔍 Offensive content detection using trained classifier")

if clf and input_text.strip():
    emb = sbert_model.encode([input_text])
    pred = clf.predict(emb)[0]
    proba = clf.predict_proba(emb)[0][pred]
    if pred == 1:
        st.error(f"❌ Offensive content detected by trained classifier (confidence: {proba:.2f})")
    else:
        st.success(f"✅ Text is not offensive by trained classifier (confidence: {proba:.2f})")
elif input_text.strip():
    st.info("ℹ️ Not enough training data to use trained classifier. Use pretrained model or add examples.")

