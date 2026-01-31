# rag_pipeline.py
import faiss, numpy as np, pandas as pd, os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---------- CONFIG ----------
DATA_DIR = "data"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMB_FILE = os.path.join(DATA_DIR, "embeddings.npy")
INDEX_FILE = os.path.join(DATA_DIR, "reviews.index")
DF_FILE = os.path.join(DATA_DIR, "reviews_df.pkl")

# ---------- LOAD ----------
df = pd.read_pickle(DF_FILE)
index = faiss.read_index(INDEX_FILE)
embeddings = np.load(EMB_FILE).astype("float32")

model = SentenceTransformer(MODEL_NAME)

genai.configure(api_key="AIzaSyCOSo1dDD5iQucfPfGEXSkizly5N-rb0YA")
llm = genai.GenerativeModel("models/gemini-2.5-flash")

# ---------- FUNCTIONS ----------
def retrieve(query, k=7):
    q_emb = model.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        row = df.iloc[idx]
        results.append({
            "review_text": row["Review"],
            "rating": row.get("Rating", "")
        })
    return results

def gemini_summary(query, retrieved):
    context = "\n".join(
        [f"- {r['review_text']} (Rating: {r['rating']})" for r in retrieved]
    )

    prompt = f"""
You are an expert customer feedback analyst.

User Query:
{query}

Relevant Customer Reviews:
{context}

Generate:
1. A short executive summary (2–3 sentences)
2. Key pain points
3. Actionable recommendation
"""

    response = llm.generate_content(prompt)
    return response.text.strip()
