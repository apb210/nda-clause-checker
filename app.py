from docx import Document
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load model once
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

def extract_clauses(docx_file):
    doc = Document(docx_file)
    clause_pattern = re.compile(r'^(\d{1,2})\.\s+(.*)')
    clauses = {}
    current_number = None
    buffer = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        match = clause_pattern.match(text)
        if match:
            if current_number:
                clauses[current_number] = ' '.join(buffer).strip()
            current_number = match.group(1)
            buffer = [match.group(2)]
        elif current_number:
            buffer.append(text)

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
            if sim_score > 0.65:  # Adjust this threshold if needed
                match_found = True
                similarity_results[exp_key] = (sim_score, ext_key, ext_text)
                break
        if not match_found:
            missing_clauses.append(exp_key)
            similarity_results[exp_key] = (0.0, None, None)

    return similarity_results, missing_clauses

def main():
    st.title("📄 NDA Clause Auditor with NLP")

    uploaded_file = st.file_uploader("Upload your NDA (.docx)", type="docx")
    if uploaded_file:
        st.info("🔍 Extracting clauses...")
        extracted = extract_clauses(uploaded_file)

        st.subheader("✅ Extracted Clauses")
        for num, text in extracted.items():
            st.markdown(f"**Clause {num}:** {text}")

        st.subheader("🤖 Comparing Using NLP")
        similarities, missing = nlp_compare(extracted, EXPECTED_CLAUSES)

        for num in sorted(EXPECTED_CLAUSES.keys(), key=int):
            summary = EXPECTED_CLAUSES[num]
            sim_score, matched_clause, text = similarities[num]
            if sim_score == 0.0:
                st.error(f"❌ Missing Clause {num}: {summary}")
            else:
                st.success(f"✅ Clause {num} matched (Score: {sim_score:.2f})")

        if not missing:
            st.success("🎉 All expected clauses were contextually identified!")

if __name__ == "__main__":
    main()
