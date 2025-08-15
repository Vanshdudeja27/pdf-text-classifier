# import streamlit as st

import nltk
import streamlit as st
import os



# Set a persistent data path
data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
nltk.data.path.append(data_dir)

# Download NLTK data if not present
required_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']

for data_item in required_data:
    try:
        nltk.data.find(f'tokenizers/{data_item}') # Example for punkt
    except LookupError:
        # st.info(f"Downloading NLTK resource: {data_item}...")
        nltk.download(data_item, download_dir=data_dir)
        # st.success(f"{data_item} downloaded successfully!")

# Create a data directory if it doesn't exist


import PyPDF2
import re
# import nltk
import joblib


# Import necessary NLTK components for text processing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Downloads
# These lines ensure that the required NLTK data is available.
# NLTK will check if the resources are already downloaded before trying to get them again.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --- Load the pre-trained model and vectorizer ---
# The app relies on these two files, which must be in the same directory.
# 'pdf_text_classifier_model.pkl' contains the trained classification model.
# 'tfidf_vectorizer.pkl' contains the TF-IDF vectorizer fitted on the training data.
try:
    model = joblib.load("pdf_text_classifier_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}. Please ensure 'pdf_text_classifier_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()


# --- NLP Tools Setup ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# --- Preprocessing Function ---
# This function cleans and tokenizes the input text.
def preprocess(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
    tokens = nltk.word_tokenize(text) # Tokenize into words
    # Filter out non-alphabetic tokens and stopwords
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    # Apply stemming and lemmatization to normalize the words
    tokens = [lemmatizer.lemmatize(stemmer.stem(t)) for t in tokens]
    return ' '.join(tokens)

# --- PDF Text Extraction Function ---
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        # Iterate through pages and join the text
        return " ".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        st.error(f"Failed to read the PDF file: {e}")
        return ""

# --- Label Mapping ---
label_map = {0: "invoice", 1: "report", 2: "resume"}

# --- Streamlit UI: Page Configuration ---
st.set_page_config(page_title="📄 PDF Classifier", layout="centered", page_icon="📎")

# --- Streamlit UI: Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ App Info")
    st.info("Upload a PDF and get its predicted category.\n\nSupports **Invoice**, **Report**, and **Resume**.\n\nModel: `TF-IDF + Self-training`")

# --- Streamlit UI: Custom CSS ---
# This adds some styling to the app for a better user experience.
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

# --- Streamlit UI: Main Content ---
st.markdown("<h1 style='text-align: center;'>📄 PDF Text Classifier</h1>", unsafe_allow_html=True)
st.caption("Upload a PDF to predict if it's an *invoice*, *report*, or *resume*.")

# --- File Uploader Widget ---
uploaded_file = st.file_uploader("📥 Upload a PDF file", type="pdf")

if uploaded_file:
    # Use a spinner to provide visual feedback to the user
    with st.spinner("🔍 Extracting and analyzing text..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        word_count = len(raw_text.split())

        if len(raw_text.strip()) < 30:
            st.warning("⚠️ Not enough text found in the PDF.")
        else:
            # Main logic for prediction
            clean_text = preprocess(raw_text)
            vector = tfidf.transform([clean_text])
            prediction = model.predict(vector)[0]
            label = label_map.get(prediction, "Unknown")

            # Display the results
            st.markdown(f"### ✅ Predicted Category: <span style='color:lightgreen'>{label.capitalize()}</span>", unsafe_allow_html=True)
            st.markdown(f"📝 **Words Extracted:** `{word_count}`")

            # Expanders to show the raw and preprocessed text
            with st.expander("📘 Extracted Text (first 500 chars)", expanded=True):
                st.text_area("Preview:", raw_text[:500], height=200)

            with st.expander("🧠 Preprocessed Tokens"):
                st.write(clean_text)
