
import pandas as pd
import re
import unicodedata
from bs4 import BeautifulSoup
import wordninja
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore", message=".*token_pattern.*")
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ibpipeline
from imblearn.over_sampling import SMOTE

# 1) Thiết lập NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 2) Đọc dữ liệu (chứa cột "text", các categorical và numeric như đã thỏa thuận)
df = pd.read_csv("C:\\Users\\DELL 15\\Documents\\Zalo Received Files\\data_processed.csv")  # sửa đường dẫn nếu cần
y = df["fraudulent"]
X = df.drop(columns=["fraudulent"])
text_col = "text"
smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X, y)
# 1. Tách dữ liệu huấn luyện & kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# # 3. Huấn luyện SVM với class_weight='balanced'
# svm = SVC(kernel='linear', class_weight='balanced')
# svm.fit(X_resampled, y_resampled)

# # 4. Dự đoán và đánh giá
# y_pred = svm.predict(X_test)
# print(classification_report(y_test, y_pred))

categorical_cols = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
numeric_cols = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "missing_employment_type",
    "missing_benefits",
    "missing_description",
    "missing_company_profile",
    "missing_location",
    "missing_department",
]

# 3) Hàm tiền xử lý văn bản
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

# 4) Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Định nghĩa preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("tfidf", TfidfVectorizer(
         max_features=10000,
         tokenizer=nlp_process,
         ngram_range=(1,2)
     ), text_col),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

# 6) Xây pipeline với SVM
pipeline_svm = ibpipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        random_state=42,
        verbose=1,
        class_weight="balanced"
    )),
])

# 7) Huấn luyện & Đánh giá
print("Training SVM pipeline...")
pipeline_svm.fit(X_train, y_train)

print("Evaluating on test set...")
y_pred = pipeline_svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
