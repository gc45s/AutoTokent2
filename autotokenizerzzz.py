import pandas as pd
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Load Sentence BERT model
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Example input data
idiom_data = pd.DataFrame({
    "Idiom": ["Break a leg", "猫の手も借りたい"],
    "Language": ["English", "Japanese"]
})

LANG_CODE_MAP = {
    "English": "en",
    "Indonesian": "id",
    "Japanese": "ja",
    "Thai": "th",
    "Filipino": "tl"
}

def translate_idiom(idiom, target_lang_code):
    try:
        return GoogleTranslator(source='auto', target=target_lang_code).translate(idiom)
    except Exception:
        return "(Translation failed)"

def validate_and_reason(df):
    results = []
    for _, row in df.iterrows():
        idiom = row['Idiom']
        lang = row['Language']
        lang_code = LANG_CODE_MAP.get(lang, 'en')

        meaning = translate_idiom(idiom, lang_code)
        idiom_emb = sbert.encode(idiom, convert_to_tensor=True)
        context_emb = sbert.encode(f"Common idioms in {lang}", convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(idiom_emb, context_emb).item()

        valid = 1 if similarity > 0.3 else -1
        reason = (f"Idiom '{idiom}' is commonly associated with the {lang} language. "
                  f"Detected similarity score: {similarity:.2f}. Meaning: '{meaning}'.")

        results.append({
            "Language": lang,
            "Idiom": idiom,
            "Meaning": meaning,
            "Similarity": similarity,
            "Validated": valid,
            "Reason": reason
        })

    return pd.DataFrame(results)

validated_df = validate_and_reason(idiom_data)
print(validated_df)

# Optionally save
validated_df.to_csv("validated_idioms.csv", index=False)
