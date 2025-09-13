import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
import nltk
import wordninja
import re
from bs4 import BeautifulSoup
import unicodedata
import sys
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns

# Tải các dữ liệu cần thiết từ NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Đọc dữ liệu
df = pd.read_csv("C:\\Users\\DELL 15\\Documents\\Zalo Received Files\\data_processed.csv")
print(df["fraudulent"].value_counts())
# Thêm flag contains_scam_keywords nếu văn bản có từ nghi ngờ
import pandas as pd

# Đọc lại file data_processed.csv
df = pd.read_csv("C:\\Users\\DELL 15\\Documents\\Zalo Received Files\\data_processed.csv")

# Danh sách đầy đủ từ khóa scam (đã chuẩn hóa viết thường)
scam_keywords = [
    "pay to apply", "registration fee", "application fee", "training fee", "sign up fee", "pre-employment fee",
    "deposit required", "send payment", "verification fee", "upfront fee", "transfer fee",
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

# Hàm xử lý kiểm tra có từ scam không
def contains_scam_keyword(text):
    text = str(text).lower()
    return int(any(keyword in text for keyword in scam_keywords))

# Tạo cột mới
df["contains_scam_keywords"] = df["text"].apply(contains_scam_keyword)

# Xuất file mới để huấn luyện
processed_output = "C:\\Users\\DELL 15\\Documents\\Zalo Received Files\\data_processed.csv"
df.to_csv(processed_output, index=False)

processed_output


text_col = "text"
categorical_cols = ['employment_type', 'required_experience', 'required_education',
                   'industry', 'function']
numeric_cols = ['telecommuting', 'has_company_logo', 'has_questions',
               'missing_employment_type', 'missing_benefits', 'missing_description',
               'missing_company_profile', 'missing_location', 'missing_department']

X = df.drop(columns=["fraudulent"])
y = df["fraudulent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

# ===== preprocessor (drop-in) =====
preprocessor = ColumnTransformer(
    transformers=[
        ('text',
         TfidfVectorizer(
             max_features=50000,
             stop_words='english',         
             tokenizer=nlp_process,
             ngram_range=(1, 3),           
             sublinear_tf=True,
             min_df=3
         ),
         text_col),
        ('cat',
         OneHotEncoder(handle_unknown='ignore', sparse_output=True),  
         categorical_cols),
        ('num',
         StandardScaler(with_mean=False),
         numeric_cols)
    ]
)

# ===== pipeline (drop-in) =====
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

pipeline = ImbPipeline(steps=[
    ('preprocess', preprocessor),
    # Thử BorderlineSMOTE giúp học tốt biên quyết định; nếu muốn giữ SMOTE cũ thì thay lại.
    ('smote', BorderlineSMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)),
    ('model', LogisticRegression(max_iter=30000, solver='liblinear',
                                 C=5.0, class_weight=None, random_state=42))
])


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nModel Evaluation after SMOTE:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

dump(pipeline, 'fraud_detection_model_smote.pkl')
print("\nModel with SMOTE saved as 'fraud_detection_model_smote.pkl'")