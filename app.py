# app.py
import streamlit as st
import tempfile
import os
from io import BytesIO
import PyPDF2
import docx2txt
import re
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Hugging Face transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers.pipelines import PipelineException

st.set_page_config(layout="wide", page_title="ContractIQ - Prototype")

# ---------------------------
# Helper: Text extraction
# ---------------------------
def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            try:
                text.append(page.extract_text() or "")
            except:
                text.append("")
    return "\n".join(text)

def extract_text_from_docx(file_path):
    try:
        return docx2txt.process(file_path)
    except Exception:
        return ""

# ---------------------------
# Helper: Chunking / clause segmentation
# ---------------------------
def split_into_clauses(text, max_clause_sentences=6):
    # Normalize whitespace
    t = re.sub(r'\r', '\n', text)
    t = re.sub(r'\n{2,}', '\n\n', t)
    # Split by common legal headings or newlines
    # First try to split on headings like "1. Termination", "Termination:", "Confidentiality"
    heading_regex = r'(?m)^(?:\d+\.\s*)?[A-Z][A-Za-z \-\(\)\/]{3,50}\s*$'
    lines = t.splitlines()
    clauses = []
    current = []
    for line in lines:
        if re.match(heading_regex, line.strip()):
            if current:
                clauses.append("\n".join(current).strip())
                current = []
            current.append(line.strip())
        else:
            current.append(line)
    if current:
        clauses.append("\n".join(current).strip())

    # If too coarse, further split by paragraphs
    refined = []
    for c in clauses:
        paragraphs = [p.strip() for p in re.split(r'\n{1,}', c) if p.strip()]
        # merge small paragraphs
        buffer = []
        for p in paragraphs:
            buffer.append(p)
            if len(" ".join(buffer).split('.')) >= max_clause_sentences:
                refined.append(" ".join(buffer).strip())
                buffer = []
        if buffer:
            refined.append(" ".join(buffer).strip())
    # fallback if nothing found
    if not refined:
        refined = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()][:30]
    # Limit to first 200 clauses to keep pipelines manageable
    return refined[:200]

# ---------------------------
# AI pipelines (with graceful fallback)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    errors = []
    # Summarization model
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        models['summarizer'] = summarizer
    except Exception as e:
        errors.append(f"summarizer:{e}")
        models['summarizer'] = None

    # Zero-shot classification model (distil-mnli style)
    try:
        classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
        models['classifier'] = classifier
    except Exception as e:
        errors.append(f"classifier:{e}")
        models['classifier'] = None

    return models, errors

# ---------------------------
# Zero-shot classification labels
# ---------------------------
CLAUSE_LABELS = [
    "Termination", "Confidentiality", "Liability", "Payment", "Governing Law",
    "Intellectual Property", "Warranty", "Indemnity", "Force Majeure", "Privacy",
    "Definitions", "Scope of Work", "Assignment", "Notice", "Dispute Resolution"
]

# ---------------------------
# Keyword fallback classification
# ---------------------------
KEYWORD_MAP = {
    "Termination": ["terminate", "termination", "end of contract", "cancel", "expiry", "breach"],
    "Confidentiality": ["confidential", "non-disclosure", "nda", "secret", "proprietary"],
    "Liability": ["liability", "liable", "loss", "damage", "hold harmless"],
    "Payment": ["payment", "invoice", "due", "fee", "charges", "remit"],
    "Governing Law": ["governed by", "jurisdiction", "governing law", "court"],
    "Intellectual Property": ["intellectual property", "ip", "copyright", "patent", "trademark"],
    "Warranty": ["warranty", "warranties", "guarantee", "represent"],
    "Indemnity": ["indemnify", "indemnity", "hold harmless"],
    "Force Majeure": ["force majeure", "act of god", "unforeseeable"],
    "Privacy": ["personal data", "privacy", "gdpr", "data protection"],
    "Definitions": ["means", "definition", "defined terms"],
    "Scope of Work": ["scope", "services", "deliverable"],
    "Assignment": ["assign", "assignment", "transfer"],
    "Notice": ["notice", "notify", "notification"],
    "Dispute Resolution": ["dispute", "arbitration", "mediation", "litigation"]
}

def keyword_classify(text):
    scores = {}
    t = text.lower()
    for label, kws in KEYWORD_MAP.items():
        score = sum(1 for kw in kws if kw in t)
        scores[label] = score
    # produce a sorted list of labels with positive scores, else 'Definitions'
    sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_label = sorted_labels[0][0]
    return top_label, scores

# ---------------------------
# Risk scoring heuristic
# ---------------------------
def heuristic_risk_for_clause(text, label):
    t = text.lower()
    risk = 0
    # High risk keywords
    high_risk = ["penalty", "liability", "indemnify", "breach", "terminate for convenience", "liquidated damages", "hold harmless", "exclusive", "mandatory arbitration", "no liability"]
    medium_risk = ["payment", "fee", "notice", "warranty", "limitations", "confidential", "royalty", "governing law"]
    low_risk = ["definitions", "scope", "notice", "assignment"]
    for kw in high_risk:
        if kw in t:
            risk += 3
    for kw in medium_risk:
        if kw in t:
            risk += 2
    for kw in low_risk:
        if kw in t:
            risk += 1
    # label-based tweaks
    if label in ["Liability", "Indemnity", "Termination", "Payment"]:
        risk += 2
    if label in ["Confidentiality", "Privacy"]:
        risk += 1
    # Normalize to 0-10
    risk = min(risk, 10)
    # Map to category
    if risk >= 6:
        cat = "High"
    elif risk >= 3:
        cat = "Medium"
    else:
        cat = "Low"
    # Numeric score 0-100
    numeric = int((risk / 10) * 100)
    return numeric, cat

# ---------------------------
# PDF report generation
# ---------------------------
def create_pdf_report(summary, clause_rows, overall_risk, filename="contract_report.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    w, h = letter
    x_margin = 50
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, "ContractIQ - AI Risk Report")
    c.setFont("Helvetica", 10)
    y -= 25
    c.drawString(x_margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, f"Overall Risk Score: {overall_risk}%")
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x_margin, y, "Summary:")
    y -= 14
    c.setFont("Helvetica", 10)
    for line in summary.splitlines():
        for chunk in split_line(line, 90):
            if y < 80:
                c.showPage()
                y = h - 50
            c.drawString(x_margin, y, chunk)
            y -= 12
    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x_margin, y, "Clauses & Risk:")
    y -= 14
    c.setFont("Helvetica", 10)
    for r in clause_rows:
        text_line = f"- [{r['RiskCategory']}] {r['Label']}: {truncate(r['Clause'], 120)}"
        for chunk in split_line(text_line, 90):
            if y < 80:
                c.showPage()
                y = h - 50
            c.drawString(x_margin, y, chunk)
            y -= 12
        y -= 2
    c.save()
    buffer.seek(0)
    return buffer

def split_line(s, n):
    # split s into chunks of approx n chars at whitespace
    words = s.split()
    chunks = []
    cur = ""
    for w in words:
        if len(cur) + 1 + len(w) <= n:
            cur = (cur + " " + w).strip()
        else:
            chunks.append(cur)
            cur = w
    if cur:
        chunks.append(cur)
    return chunks

def truncate(s, n):
    return (s[:n] + "...") if len(s) > n else s

# ---------------------------
# Main analysis pipeline
# ---------------------------
def analyze_contract(text, models, top_k=3):
    clauses = split_into_clauses(text)
    classifier = models.get('classifier')
    summarizer = models.get('summarizer')

    results = []
    batch = clauses

    # Classify clauses (try zero-shot; fallback to keywords)
    if classifier is not None:
        try:
            # The zero-shot classifier can receive a list; we'll classify each clause
            for c in batch:
                try:
                    res = classifier(c, candidate_labels=CLAUSE_LABELS, multi_label=False)
                    label = res['labels'][0]
                    score = float(res['scores'][0])
                except PipelineException as e:
                    label, _ = keyword_classify(c)
                    score = 0.0
                numeric, cat = heuristic_risk_for_clause(c, label)
                results.append({
                    "Clause": c,
                    "Label": label,
                    "Confidence": float(score),
                    "RiskScore": numeric,
                    "RiskCategory": cat
                })
        except Exception as e:
            # fallback
            for c in batch:
                label, _ = keyword_classify(c)
                numeric, cat = heuristic_risk_for_clause(c, label)
                results.append({
                    "Clause": c,
                    "Label": label,
                    "Confidence": 0.0,
                    "RiskScore": numeric,
                    "RiskCategory": cat
                })
    else:
        # pure keyword fallback
        for c in batch:
            label, _ = keyword_classify(c)
            numeric, cat = heuristic_risk_for_clause(c, label)
            results.append({
                "Clause": c,
                "Label": label,
                "Confidence": 0.0,
                "RiskScore": numeric,
                "RiskCategory": cat
            })

    # Summarization
    summary_text = ""
    if summarizer is not None:
        try:
            # summarizer expects not too long input; give it first ~1500 tokens of text
            short_input = " ".join(text.split()[:1500])
            sumres = summarizer(short_input, max_length=150, min_length=40, do_sample=False)
            summary_text = sumres[0]['summary_text']
        except Exception as e:
            # fallback simple first N lines
            summary_text = " ".join(text.splitlines()[:5])
    else:
        summary_text = " ".join(text.splitlines()[:5])

    # Overall risk
    if results:
        overall = int(sum([r['RiskScore'] for r in results]) / len(results))
    else:
        overall = 0

    return results, summary_text, overall

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title("ContractIQ — Prototype (Semi-real AI)")
    st.markdown("""
    Upload a contract (PDF / DOCX). The prototype will:
    - extract clauses, 
    - classify them into clause-types (zero-shot), 
    - assign heuristic risk scores (Low / Medium / High), 
    - provide a short plain-English summary, and
    - let you download a PDF risk report.
    """)
    models, load_errors = load_models()
    if load_errors:
        st.info("Model load notice: some models may not have loaded. The app will fallback to keyword-based processing if needed.")

    uploaded = st.file_uploader("Upload contract (PDF or DOCX)", type=['pdf', 'docx'], accept_multiple_files=False)
    max_clauses_display = st.sidebar.slider("Max clauses to display", 5, 50, 25)
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_ext = uploaded.name.split('.')[-1].lower()

        with st.spinner("Extracting text..."):
            if file_ext == "pdf":
                text = extract_text_from_pdf(file_bytes)
            else:
                # write to temp file for docx
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                tmp.write(file_bytes)
                tmp.flush()
                tmp.close()
                text = extract_text_from_docx(tmp.name)
                try:
                    os.unlink(tmp.name)
                except:
                    pass

        if not text.strip():
            st.error("No text could be extracted from the uploaded file.")
            st.stop()

        st.subheader("Raw extracted text (first 8000 chars)")
        st.text_area("Extracted Text", value=text[:8000], height=220)

        with st.spinner("Analyzing contract (this may download models on first run)..."):
            clause_results, summary_text, overall = analyze_contract(text, models)
        st.success(f"Analysis complete — Overall risk score: {overall}%")

        # Show summary + overall risk graphic
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Plain-English Summary")
            st.write(summary_text)
        with col2:
            st.subheader("Overall Risk")
            fig, ax = plt.subplots(figsize=(3,2))
            ax.barh([0], [overall], height=0.6)
            ax.set_xlim(0,100)
            ax.set_yticks([])
            ax.set_xlabel("Risk %")
            ax.set_title(f"Overall: {overall}%")
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)

        # Show clauses table
        df = pd.DataFrame(clause_results)
        if df.empty:
            st.warning("No clauses were detected.")
            st.stop()

        df_display = df[['Label','Confidence','RiskScore','RiskCategory','Clause']].head(max_clauses_display)
        st.subheader(f"Top {len(df_display)} detected clauses")
        def color_row(r):
            if r['RiskCategory'] == "High":
                return ['background-color: #ffd6d6']*5
            elif r['RiskCategory'] == "Medium":
                return ['background-color: #fff2cc']*5
            else:
                return ['background-color: #e6ffea']*5
        st.dataframe(df_display.reset_index(drop=True).style.apply(color_row, axis=1), height=400)

        # Expand per-clause viewers
        st.subheader("Clauses (detailed)")
        for i, row in enumerate(clause_results[:max_clauses_display]):
            box = st.expander(f"[{row['RiskCategory']}] {row['Label']} — Score {row['RiskScore']} — Conf {row['Confidence']:.2f}")
            with box:
                st.write(row['Clause'])

        # Downloadable report
        if st.button("Generate & Download PDF Risk Report"):
            with st.spinner("Generating PDF..."):
                pdf_buf = create_pdf_report(summary_text, clause_results[:max_clauses_display], overall, filename="contract_report.pdf")
                st.download_button("Download Report", data=pdf_buf, file_name="contract_risk_report.pdf", mime="application/pdf")

        # Offer CSV of clause data
        csv = pd.DataFrame(clause_results).to_csv(index=False).encode('utf-8')
        st.download_button("Download clause data (CSV)", data=csv, file_name="clauses.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("Prototype by Team NYD — ContractIQ (CodeSynthsis).")

if __name__ == "__main__":
    main()
