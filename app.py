import streamlit as st
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import os

# --- Resume Parsing Functions ---
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def extract_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".pdf":
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return extract_text_from_pdf("temp.pdf")
    elif ext == ".docx":
        with open("temp.docx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return extract_text_from_docx("temp.docx")
    else:
        raise ValueError("Only PDF or DOCX supported")

# --- ATS scoring ---
def ats_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def keyword_match(resume_text, jd_keywords):
    results = {}
    for word in jd_keywords:
        results[word] = fuzz.partial_ratio(word.lower(), resume_text.lower())
    return results

# --- Streamlit UI ---
st.title("📄 ATS Resume Checker")
st.write("Upload your resume and paste a job description to see your ATS score.")

resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description")

if resume_file and job_description:
    resume_text = extract_text(resume_file)
    score = ats_score(resume_text, job_description)

    st.subheader("✅ ATS Score")
    st.write(f"Your resume matches the job description with a score of **{score}%**")

    st.subheader("🔑 Keyword Matches")
    keywords = ["Python", "Machine Learning", "SQL", "Communication"]
    matches = keyword_match(resume_text, keywords)
    st.write(matches)
