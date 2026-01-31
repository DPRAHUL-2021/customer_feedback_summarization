# # app.py
# import streamlit as st
# from rag_pipeline import retrieve, gemini_summary

# st.set_page_config(
#     page_title="Customer Feedback Analyzer",
#     layout="wide"
# )

# st.title("🔍 Customer Feedback Summarization using RAG + LLM")
# st.write("Analyze customer reviews using semantic retrieval and AI-based summarization.")

# query = st.text_input(
#     "Enter your query",
#     placeholder="e.g. Why are offline downloads not working?"
# )

# if st.button("Analyze"):
#     if not query.strip():
#         st.warning("Please enter a query.")
#     else:
#         with st.spinner("Retrieving relevant reviews..."):
#             retrieved = retrieve(query)

#         st.subheader("📌 Retrieved Reviews (RAG Evidence)")
#         for i, r in enumerate(retrieved, 1):
#             with st.expander(f"Review {i} | Rating: {r['rating']}★"):
#                 st.write(r["review_text"])

#         with st.spinner("Generating AI summary..."):
#             summary = gemini_summary(query, retrieved)

#         st.subheader("🧠 AI-Generated Summary")
#         st.success(summary)


import streamlit as st
import time
from rag_pipeline import retrieve, gemini_summary

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Feedback Intelligence",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    color: #6c757d;
    text-align: center;
    margin-bottom: 30px;
}
.section-box {
    background-color: #f8f9fa;
    padding: 18px;
    border-radius: 10px;
}
.metric {
    background-color: #ffffff;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 0 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Customer Feedback Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Augmented Generation (RAG) powered by Large Language Models</div>',
    unsafe_allow_html=True
)

# ---------------- QUERY INPUT ----------------
query = st.text_input(
    "**Ask a question about customer feedback**",
    placeholder="Why are users complaining about offline downloads?"
)

# Inline slider (NO sidebar)
# top_k = st.slider(
#     "Number of reviews to retrieve",
#     min_value=3,
#     max_value=10,
#     value=7
# )

# col_k1, col_k2, col_k3 = st.columns([4, 1, 4])

# with col_k2:
#     top_k = st.number_input(
#         "Reviews",
#         min_value=3,
#         max_value=10,
#         value=7,
#         step=1,
#         label_visibility="collapsed"
#     )

# st.caption("Number of reviews to retrieve")

# ---------- TOP-K COMPACT INPUT ----------
# ---------- TOP-K COMPACT INPUT (LABEL ON TOP) ----------
k_col1, k_col2, k_col3 = st.columns([1.2, 1, 6])

with k_col1:
    st.markdown("**Top-K**")
    top_k = st.number_input(
        label="Top-K",
        min_value=3,
        max_value=10,
        value=7,
        step=1,
        label_visibility="collapsed"
    )





analyze = st.button("🚀 Analyze Feedback")

st.divider()

# ---------------- MAIN LOGIC ----------------
if analyze:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        start = time.time()

        with st.spinner("Retrieving relevant customer reviews..."):
            retrieved = retrieve(query, k=top_k)

        with st.spinner("Generating AI insights..."):
            summary = gemini_summary(query, retrieved)

        total_time = time.time() - start

        # ---------------- METRICS ----------------
        m1, m2, m3 = st.columns(3)
        m1.markdown('<div class="metric"><b>Reviews Retrieved</b><br>' + str(len(retrieved)) + '</div>', unsafe_allow_html=True)
        m2.markdown('<div class="metric"><b>Response Time</b><br>' + f"{total_time:.2f}s" + '</div>', unsafe_allow_html=True)
        m3.markdown('<div class="metric"><b>Pipeline</b><br>RAG + LLM</div>', unsafe_allow_html=True)

        st.divider()

        # ---------------- OUTPUT SECTIONS ----------------
        left, right = st.columns([1.3, 1.7])

        # -------- LEFT: EVIDENCE --------
        with left:
            st.markdown("### 🔍 Retrieved Evidence")
            st.caption("These reviews form the factual basis for the AI summary.")

            for i, r in enumerate(retrieved, 1):
                with st.expander(f"Review {i} | Rating: {r['rating']}★"):
                    st.write(r["review_text"])

        # -------- RIGHT: SUMMARY --------
        with right:
            st.markdown("### 🧠 AI-Generated Insights")
            st.markdown(
                f"<div class='section-box'>{summary}</div>",
                unsafe_allow_html=True
            )

        st.divider()
        st.caption("Embeddings: MPNet | Retrieval: FAISS | LLM: Gemini 2.5 Flash")
