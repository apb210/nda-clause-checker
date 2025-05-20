import os
import re
import json
import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
from io import StringIO
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

# Load LegalBERT tokenizer and model
@st.cache_resource
def load_legalbert():
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    return tokenizer, model

tokenizer, legalbert_model = load_legalbert()

# Adjustable truncation length
truncate_length = st.sidebar.slider("Truncate clause length (characters)", min_value=200, max_value=2000, value=800, step=100)

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

# Compute embedding using LegalBERT
def embed_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = legalbert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
    return embeddings

# Compare clauses
def compare_clauses(extracted_clauses, standard_clauses):
    results = []
    for label, std_clause in standard_clauses.items():
        best_match = None
        best_score = -1
        std_embedding = embed_text(std_clause)
        for clause in extracted_clauses:
            if len(clause.strip()) < 20:
                continue
            clause_embedding = embed_text(clause.strip())
            score = cosine_similarity(std_embedding, clause_embedding).item()
            if score > best_score:
                best_score = score
                best_match = clause.strip()
        if best_score == -1:
            results.append((label, std_clause, "Clause not found", 0.0))
        else:
            truncated_match = best_match[:truncate_length] + ("..." if len(best_match) > truncate_length else "")
            results.append((label, std_clause, truncated_match, round(best_score, 3)))
    return results

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
    results = compare_clauses(clauses, standard_clauses)

    st.subheader("Clause Comparison Report")
    data = []
    for label, std, match, score in results:
        st.markdown(f"**{label}**")
        with st.expander("View Standard Clause"):
            st.markdown(std)
        with st.expander("View Matched Clause"):
            st.markdown(match)
        st.markdown(f"**Similarity Score:** {score}")
        if match == "Clause not found" or score < 0.85:
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
