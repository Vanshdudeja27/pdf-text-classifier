import streamlit as st
import PyPDF2
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("pdf_text_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# NLP tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(stemmer.stem(t)) for t in tokens]
    return ' '.join(tokens)

# PDF text extraction
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() or "" for page in reader.pages])

# Label map
label_map = {0: "invoice", 1: "report", 2: "resume"}

# --- Page Setup ---
st.set_page_config(page_title="📄 PDF Classifier", layout="centered", page_icon="📎")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ App Info")
    st.info("Upload a PDF and get its predicted category.\n\nSupports **Invoice**, **Report**, and **Resume**.\n\nModel: `TF-IDF + Self-training`")

# --- Custom CSS ---
st.markdown("""
    <style>
        .css-1aumxhk { padding-top: 2rem; }
        .stTextArea textarea { font-family: monospace; font-size: 0.9rem; }
        .reportview-container .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Main Title ---
st.markdown("<h1 style='text-align: center;'>📄 PDF Text Classifier</h1>", unsafe_allow_html=True)
st.caption("Upload a PDF to predict if it's an *invoice*, *report*, or *resume*.")

# --- File Upload ---
uploaded_file = st.file_uploader("📥 Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Extracting and analyzing text..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        word_count = len(raw_text.split())

        if len(raw_text.strip()) < 30:
            st.warning("⚠️ Not enough text found in the PDF.")
        else:
            clean_text = preprocess(raw_text)
            vector = tfidf.transform([clean_text])
            prediction = model.predict(vector)[0]
            label = label_map.get(prediction, "Unknown")

            st.markdown(f"### ✅ Predicted Category: <span style='color:lightgreen'>{label.capitalize()}</span>", unsafe_allow_html=True)
            st.markdown(f"📝 **Words Extracted:** `{word_count}`")

            # Text Display
            with st.expander("📘 Extracted Text (first 500 chars)", expanded=True):
                st.text_area("Preview:", raw_text[:500], height=200)

            with st.expander("🧠 Preprocessed Tokens"):
                st.write(clean_text)
