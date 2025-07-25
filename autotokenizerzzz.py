import streamlit as st
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
import joblib
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

#---cek sistem secara default---
def get_default_threads():
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    freq_mhz = cpu_freq.current if cpu_freq else 0

    # Cek sederhana: kalau core sedikit atau frekuensi rendah, set thread kecil
    if cpu_count <= 2 or freq_mhz < 2000:
        default_threads = 1
    elif cpu_count <= 4 or freq_mhz < 3000:
        default_threads = 2
    else:
        default_threads = min(4, cpu_count)  # maksimal 4 thread atau sesuai cpu_count

    return default_threads

default_threads = get_default_threads()
torch.set_num_threads(default_threads)
print(f"[INFO] Set PyTorch threads to {default_threads} based on detected CPU performance")

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

st.sidebar.title("🛍️ Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Dashboard", "🛡️ Deteksi Teks", "🧠 Analisis Idiom", "🗂️ Manajemen Data"])

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

elif page == "🧠 Analisis Idiom":
    st.markdown("## 🧠 Analisis Idiom Berdasarkan Data Input")

    st.info("Masukkan idiom dan pilih bahasanya, lalu klik **Analisis Idiom**. Kolom 'Meaning' akan diterjemahkan otomatis.")

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

    if st.button("🔍 Analisis Idiom"):
        with st.spinner("Menghitung kemiripan dan menerjemahkan..."):
            try:
                results = []
                similarity_scores = []

                for _, row in idiom_input_df.iterrows():
                    lang = row["Language"]
                    idiom = row["Idiom"]

                    if not lang or not idiom:
                        continue

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

                    lang_context = f"Common idioms in {lang}"
                    idiom_emb = sbert.encode(idiom, convert_to_tensor=True)
                    lang_emb = sbert.encode(lang_context, convert_to_tensor=True)
                    sim = util.pytorch_cos_sim(idiom_emb, lang_emb)
                    sim_score = sim.item()
                    similarity_scores.append(sim_score)

                    try:
                        encoded_ex = tokenizer(meaning, return_tensors="pt", truncation=True)
                        decoded = tokenizer.decode(encoded_ex['input_ids'][0], skip_special_tokens=True)
                        example = decoded.capitalize() + "."
                    except:
                        example = "(No example generated)"

                    results.append({
                        "Language": lang,
                        "Idiom": idiom,
                        "Meaning": meaning,
                        "Similarity": round(sim_score, 3),
                        "Example": example
                    })

                if results:
                    df_idiom_result = pd.DataFrame(results)
                    average_sim = np.mean(similarity_scores) if similarity_scores else 0
                    thresholds = {
                        "valid": average_sim * 1.00,
                        "uncertain": average_sim * 0.70
                    }

                    def validate(row):
                        if row["Similarity"] >= thresholds["valid"]:
                            return 1
                        elif row["Similarity"] >= thresholds["uncertain"]:
                            return 0
                        else:
                            return -1

                    df_idiom_result["Validated"] = df_idiom_result.apply(validate, axis=1)
                    df_idiom_result["Reason"] = df_idiom_result.apply(
                        lambda row: f"'{row['Idiom']}' berarti: {row['Meaning']}. Validasi: {['❌ Tidak Valid','⚠️ Ragu-Ragu','✅ Valid Idiom'][row['Validated']+1]} (skor: {row['Similarity']:.2f})",
                        axis=1
                    )
                    df_idiom_result["Model Known Since"] = "2019"

                    st.success("✅ Analisis selesai.")
                    st.dataframe(df_idiom_result, use_container_width=True)
                    df_idiom_result.to_csv("idiom_analysis.csv", index=False)
                else:
                    st.warning("Tidak ada idiom valid untuk dianalisis.")

            except Exception as e:
                st.error(f"Gagal memuat model atau melakukan analisis: {e}")

elif page == "🗂️ Manajemen Data":
    
    st.title("🗂️ Dataset dan Model")
    st.markdown("### ✍️ Tambah Contoh Teks")
    
    with st.form("form_data"):
        input_text = st.text_input("Teks:")
        input_label = st.selectbox("Label", LABELS)
        submit = st.form_submit_button("➕ Simpan")

    if submit:
        if os.path.exists(DEFAULT_CSV): #add
            df = pd.read_csv(DEFAULT_CSV) 
            st.success(f"Menampilkan {len(df)} data yang telah ditambahkan.")
            st.dataframe(df, use_container_width=True)
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
