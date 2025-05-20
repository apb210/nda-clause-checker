import os
import re
import json
import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sentence_transformers import SentenceTransformer, util
import hashlib

# Load faster sentence-transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# Adjustable truncation and similarity threshold
truncate_length = st.sidebar.slider("Truncate clause length (characters)", min_value=200, max_value=2000, value=800, step=100)
similarity_threshold = st.sidebar.slider("Minimum similarity threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

# Embedding cache
@st.cache_data(show_spinner=False)
def cached_embedding(text):
    return model.encode(text, convert_to_tensor=True)

# Load and parse clause sets from directory
def load_clause_set(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return parse_standard_clauses_from_txt(f, is_uploaded=False)

# Parse clauses from .txt file
def parse_standard_clauses_from_txt(file, is_uploaded=True):
    text = file.read().decode("utf-8") if is_uploaded else file.read()
    clauses = {}
    current_title = None
    current_text = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^\d+\.\s", line):
            if current_title and current_text:
                clause_text = " ".join(current_text).strip()
                clauses[current_title] = clause_text[:truncate_length] + ("..." if len(clause_text) > truncate_length else "")
            parts = line.split(".", 1)
            current_title = parts[1].strip()
            current_text = []
        elif current_title:
            current_text.append(line)
    if current_title and current_text:
        clause_text = " ".join(current_text).strip()
        clauses[current_title] = clause_text[:truncate_length] + ("..." if len(clause_text) > truncate_length else "")
    return clauses

# Extract text from PDF
@st.cache_data
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Extract text from DOCX
@st.cache_data
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract clauses
def extract_clauses(text):
    return re.split(r'\.\s+', text.strip())

# Compare clauses

def compare_clauses(extracted_clauses, standard_clauses):
    results = []
    heatmap_data = []
    for label, std_clause in standard_clauses.items():
        std_embedding = cached_embedding(std_clause)
        clause_scores = []
        for clause in extracted_clauses:
            if len(clause.strip()) < 20:
                continue
            clause_embedding = cached_embedding(clause.strip())
            score = util.pytorch_cos_sim(std_embedding, clause_embedding).item()
            clause_scores.append((clause, score))
        if clause_scores:
            best_match, best_score = max(clause_scores, key=lambda x: x[1])
            truncated_match = best_match[:truncate_length] + ("..." if len(best_match) > truncate_length else "")
            results.append((label, std_clause, truncated_match, round(best_score, 3)))
            heatmap_data.append([label] + [round(score, 2) for _, score in clause_scores])
        else:
            results.append((label, std_clause, "Clause not found", 0.0))
            heatmap_data.append([label] + [0.0] * len(extracted_clauses))
    return results, heatmap_data, extracted_clauses

# Streamlit UI
st.title("NDA Clause Checker")

# Clause set dropdown
clause_set_options = {
    "Default": "standard_clauses.txt",
    "Alternate Set A": "alternate_clauses_a.txt",
    "Alternate Set B": "alternate_clauses_b.txt"
}

selected_clause_file = st.selectbox("Select a standard clause set", options=list(clause_set_options.keys()))

# Optional upload override
custom_clause_file = st.file_uploader("Optional: Upload Custom Standard Clauses (.txt format)", type=["txt"])

if custom_clause_file:
    standard_clauses = parse_standard_clauses_from_txt(custom_clause_file, is_uploaded=True)
else:
    clause_path = clause_set_options[selected_clause_file]
    standard_clauses = load_clause_set(clause_path)

uploaded_file = st.file_uploader("Upload an NDA (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif ext == ".docx":
        text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")

    clauses = extract_clauses(text)
    results, heatmap_data, extracted_clauses = compare_clauses(clauses, standard_clauses)

    st.subheader("Clause Comparison Report")
    data = []
    for label, std, match, score in results:
        st.markdown(f"**{label}**")
        with st.expander("View Standard Clause"):
            st.markdown(std)
        with st.expander("View Matched Clause"):
            st.markdown(match)
        st.markdown(f"**Similarity Score:** {score}")
        if match == "Clause not found" or score < similarity_threshold:
            st.warning("⚠️ FLAGGED: Missing or non-standard clause")
            status = "FLAGGED"
        else:
            st.success("✓ OK")
            status = "OK"
        data.append({"Clause Type": label, "Standard Clause": std, "Matched Clause": match, "Similarity Score": score, "Status": status})

    # Download as CSV
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    st.download_button("Download Report as CSV", data=csv, file_name="nda_clause_report.csv", mime="text/csv")

    # Display heatmap
    st.subheader("Clause Similarity Heatmap")
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data, columns=["Standard Clause"] + [f"Clause {i+1}" for i in range(len(extracted_clauses))])
        plt.figure(figsize=(min(18, 2 + len(extracted_clauses) * 0.5), min(12, 0.5 * len(standard_clauses))))
        sns.heatmap(heatmap_df.iloc[:, 1:].astype(float), annot=True, xticklabels=True, yticklabels=heatmap_df["Standard Clause"].tolist(), cmap="YlGnBu")
        st.pyplot(plt.gcf())
