#!/usr/bin/env python3
# nb_smote_full.py

import sys, re, unicodedata
import pandas as pd
import wordninja, nltk
from bs4 import BeautifulSoup

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1) NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 2) Load data
df = pd.read_csv("data_processed.csv")  # chỉnh đường dẫn nếu cần
y = df["fraudulent"]
X = df.drop(columns=["fraudulent"])
text_col = "text"

categorical_cols = [
    "employment_type","required_experience",
    "required_education","industry","function"
]
numeric_cols = [
    "telecommuting","has_company_logo","has_questions",
    "missing_employment_type","missing_benefits","missing_description",
    "missing_company_profile","missing_location","missing_department"
]

# 3) Text preprocessing functions
punctuation = dict.fromkeys(
    (i for i in range(sys.maxunicode)
     if unicodedata.category(chr(i)).startswith("P")),
    " "
)
stop_w = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def remove_url(text):
    text = re.sub(r"http?://\S+|www\.\S+", " ", text)
    return re.sub(r"#(URL|url)_.*?#", " ", text)

def remove_html(text):
    clean = re.sub(r"&[a-zA-Z0-9#]+;", "", text)
    return BeautifulSoup(clean, "html.parser").get_text()

def split_and_segment(text):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    parts = []
    for token in text.split():
        parts.extend(wordninja.split(token))
    return " ".join(parts)

def normalize_text(text):
    text = text.lower().replace(".", " ")
    text = remove_url(text)
    text = remove_html(text)
    text = split_and_segment(text)
    text = text.translate(punctuation)
    return re.sub(r'\s+', ' ', text).strip()

def nlp_process(text):
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_w]
    return [lemmatizer.lemmatize(t) for t in tokens]

# 4) Split train/test before SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Build ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("tfidf", TfidfVectorizer(
         max_features=10000,
         tokenizer=nlp_process,
         token_pattern=None,
         ngram_range=(1,2)
     ), text_col),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

# 6) Full Pipeline: preprocess → SMOTE → to_dense → GaussianNB
pipeline_nb = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("to_dense", FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
    ("model", GaussianNB()),
])

# 7) Train & Evaluate
print("Training Naive Bayes pipeline with SMOTE...")
pipeline_nb.fit(X_train, y_train)

print("Evaluating on test set...")
y_pred = pipeline_nb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
