import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import wordninja
import re
from bs4 import BeautifulSoup
import unicodedata
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# ========== CÀI ĐẶT NLTK ==========
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ========== TIỀN XỬ LÝ VĂN BẢN ==========
punctuation = dict.fromkeys((i for i in range(sys.maxunicode)
                          if unicodedata.category(chr(i)).startswith("P")), " ")

stop_w = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def remove_url(text: str) -> str:
    text = re.sub(r"http?://\S+|www\.\S+", " ", text)
    return text

def remove_component_html(text: str):
    clean_text = re.sub(r"&[a-zA-Z0-9#]+;", "", text)
    soup = BeautifulSoup(clean_text, features="html.parser")
    return soup.text

def remove_non_english_letters(text):
    return re.sub(r'[^\W\d_]', lambda m: '' if not re.match(r'[a-zA-Z]', m.group()) else m.group(), text)

def normalize_space(text):
    return re.sub(r'\s+', ' ', text).strip()

def split_and_segment_words(text):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    tokens = text.split()
    segmented = []
    for token in tokens:
        segmented.extend(wordninja.split(token))
    return ' '.join(segmented)

def nlp_process(text):
    text = text.lower()
    text = text.replace(".", " ")
    text = remove_url(text)
    text = remove_component_html(text)
    text = remove_non_english_letters(text)
    text = split_and_segment_words(text)
    text = text.translate(punctuation)
    text = normalize_space(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_w]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# ========== LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU ==========
df = pd.read_csv("C:\\Users\\DELL 15\\Documents\\Zalo Received Files\\data_processed.csv")

# Thêm tính năng scam keywords
scam_keywords = [
    "pay to apply", "registration fee", "application fee", "training fee", 
    "sign up fee", "pre-employment fee", "deposit required", "send payment",
    "verification fee", "upfront fee", "transfer fee",
    "easy money", "quick money", "fast income", "earn $1000/day", "make $500 daily",
    "income guaranteed", "get paid daily", "cash payout", "100% earnings",
    "work from home", "work from phone", "type and earn", "online earnings",
    "no experience needed", "no experience required", "no resume", "no interview", "no skills",
    "immediate start", "start today", "hiring now", "apply instantly",
    "limited slots", "first come first serve",
    "just send", "send details", "text us", "call this number", "contact recruiter directly",
    "email to apply", "no contract", "work without paperwork", "flexible illegal work",
    "not on site", "ghost recruiter", "anonymous company",
    "việc nhẹ lương cao", "thu nhập 20 triệu", "lương cực khủng",
    "không cần bằng cấp", "không cần kinh nghiệm", "chuyển khoản để nhận việc",
    "nạp tiền", "phí giữ chỗ"
]

def contains_scam_keyword(text):
    text = str(text).lower()
    return int(any(keyword in text for keyword in scam_keywords))

df["contains_scam_keywords"] = df["text"].apply(contains_scam_keyword)

# ========== CHUẨN BỊ DỮ LIỆU ==========
y = df["fraudulent"]
text_col = "text"
categorical_cols = ['employment_type', 'required_experience', 'required_education',
                   'industry', 'function']
numeric_cols = ['telecommuting', 'has_company_logo', 'has_questions',
               'missing_employment_type', 'missing_benefits', 'missing_description',
               'missing_company_profile', 'missing_location', 'missing_department',
               'contains_scam_keywords']

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

# ========== PIPELINE XỬ LÝ ==========
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=10000, stop_words='english',
                                tokenizer=nlp_process, ngram_range=(1, 2)), text_col),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# ========== RANDOM FOREST CƠ BẢN ==========
rf_basic = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("Đang huấn luyện Random Forest cơ bản...")
rf_basic.fit(X_train, y_train)

# ========== ĐÁNH GIÁ MÔ HÌNH ==========
y_pred = rf_basic.predict(X_test)

print("\n=== KẾT QUẢ RANDOM FOREST CƠ BẢN ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest Cơ Bản')
plt.show()

# Feature Importance
rf_model = rf_basic.named_steps['model']
try:
    feature_names = (
        rf_basic.named_steps['preprocess'].named_transformers_['text'].get_feature_names_out().tolist() +
        rf_basic.named_steps['preprocess'].named_transformers_['cat'].get_feature_names_out().tolist() +
        numeric_cols
    )
    
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features quan trọng nhất:")
    print(feature_importance.head(10))
    
    # Visualize top features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', 
               data=feature_importance.head(10), palette='viridis')
    plt.title('Top 10 Important Features')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("\nKhông thể trích xuất feature importance:", str(e))