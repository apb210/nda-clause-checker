import streamlit as st
import re
from io import BytesIO
from docx import Document
import pdfplumber
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

EXPECTED_CLAUSES = {
    '1': "Defines what constitutes confidential information.",
    '2': "Lists exceptions to what is considered confidential.",
    '3': "Outlines obligations to protect and not misuse confidential info.",
    '4': "Restricts disclosure to internal representatives with need-to-know.",
    '5': "Allows limited disclosure under legal orders, with notice.",
    '6': "Specifies duration of agreement and confidentiality obligations.",
    '7': "Describes return or destruction of information upon termination.",
    '8': "Requires compliance with export regulations.",
    '9': "Denies warranties on disclosed information.",
    '10': "Clarifies that no rights are transferred under the agreement.",
    '11': "Allows equitable remedies for breach.",
    '12': "Specifies how formal notices must be delivered.",
    '13': "States the relationship is independent contractors only.",
    '14': "Clarifies definitions and interpretation rules.",
    '15': "Ensures invalid provisions don’t affect rest of the agreement.",
    '16': "Explains amendment and waiver conditions.",
    '17': "Limits assignment of agreement rights and duties.",
    '18': "Establishes this as the full agreement.",
    '19': "Sets Connecticut law and courts for governing disputes.",
    '20': "Allows execution in multiple counterparts including electronic."
}

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_clauses_from_text(text):
    clause_pattern = re.compile(r'^(\d{1,2})\.\s+(.*)', re.MULTILINE)
    lines = text.splitlines()
    clauses = {}
    current_number = None
    buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = clause_pattern.match(line)
        if match:
            if current_number:
                clauses[current_number] = ' '.join(buffer).strip()
            current_number = match.group(1)
            buffer = [match.group(2)]
        elif current_number:
            buffer.append(line)

    if current_number:
        clauses[current_number] = ' '.join(buffer).strip()

    return clauses

def nlp_compare(extracted, expected):
    similarity_results = {}
    missing_clauses = []
    expected_embeddings = model.encode(list(expected.values()), convert_to_tensor=True)

    for i, (exp_key, exp_text) in enumerate(expected.items()):
        match_found = False
        for ext_key, ext_text in extracted.items():
            ext_embedding = model.encode(ext_text, convert_to_tensor=True)
            sim_score = float(util.pytorch_cos_sim(ext_embedding, expected_embeddings[i]))
            if sim_score > 0.65:
                match_found = True
                similarity_results[exp_key] = (sim_score, ext_key, ext_text)
                break
        if not match_found:
            missing_clauses.append(exp_key)
            similarity_results[exp_key] = (0.0, None, None)

    return similarity_results, missing_clauses

def main():
    st.title("📄 NDA Clause Auditor (Docx & PDF Support)")

    uploaded_file = st.file_uploader("Upload a .docx or .pdf file", type=["docx", "pdf"])
    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()

        with st.spinner("🔍 Reading document..."):
            if ext == "docx":
                text = extract_text_from_docx(uploaded_file)
            elif ext == "pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return

        clauses = extract_clauses_from_text(text)

        st.subheader("✅ Extracted Clauses")
        for k in sorted(clauses.keys(), key=int):
            st.markdown(f"**Clause {k}**: {clauses[k]}")

        st.subheader("🤖 NLP-Based Clause Comparison")
        similarities, missing = nlp_compare(clauses, EXPECTED_CLAUSES)

        for k in sorted(EXPECTED_CLAUSES.keys(), key=int):
            summary = EXPECTED_CLAUSES[k]
            sim_score, _, clause_text = similarities[k]
            if sim_score == 0.0:
                st.error(f"❌ Missing Clause {k}: {summary}")
            else:
                st.success(f"✅ Clause {k} matched (Similarity: {sim_score:.2f})")

        if not missing:
            st.balloons()

if __name__ == "__main__":
    main()
