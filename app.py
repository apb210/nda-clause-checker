import os
import re
import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
from io import StringIO, BytesIO
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Standard clauses for comparison
standard_clauses = {
    "Term Clause": "This Agreement shall terminate upon written notice or after 2 years.",
    "Return Clause": "The Receiving Party agrees to return or destroy all confidential materials upon termination.",
    "Jurisdiction Clause": "This Agreement shall be governed by the laws of the State of Delaware.",
    "Confidentiality Clause": "The Receiving Party shall not disclose any Confidential Information to third parties."
}

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
def compare_clauses(extracted_clauses):
    results = []
    for label, std_clause in standard_clauses.items():
        best_match = None
        best_score = -1
        for clause in extracted_clauses:
            if len(clause.strip()) < 20:
                continue
            emb1 = model.encode(std_clause, convert_to_tensor=True)
            emb2 = model.encode(clause.strip(), convert_to_tensor=True)
            score = util.pytorch_cos_sim(emb1, emb2).item()
            if score > best_score:
                best_score = score
                best_match = clause.strip()
        if best_score == -1:
            results.append((label, std_clause, "Clause not found", 0.0))
        else:
            results.append((label, std_clause, best_match, round(best_score, 3)))
    return results

# Generate PDF report
def generate_pdf_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="NDA Clause Comparison Report", ln=True, align="C")
    pdf.ln(10)

    for row in data:
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(200, 8, txt=f"{row['Clause Type']} ({row['Status']})", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, txt=f"Standard Clause: {row['Standard Clause']}")
        pdf.multi_cell(0, 6, txt=f"Matched Clause: {row['Matched Clause']}")
        pdf.cell(200, 6, txt=f"Similarity Score: {row['Similarity Score']}", ln=True)
        pdf.ln(4)

    buffer = BytesIO()
    pdf.output(name=buffer)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.title("NDA Clause Checker")

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
    results = compare_clauses(clauses)

    st.subheader("Clause Comparison Report")
    data = []
    for label, std, match, score in results:
        st.markdown(f"**{label}**")
        st.markdown(f"- **Standard:** {std}")
        st.markdown(f"- **Matched:** {match}")
        st.markdown(f"- **Similarity Score:** {score}")
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

    # Download as PDF
    pdf_buffer = generate_pdf_report(data)
    st.download_button("Download Report as PDF", data=pdf_buffer, file_name="nda_clause_report.pdf", mime="application/pdf")
