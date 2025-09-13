import streamlit as st
import pandas as pd
import joblib
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import wordninja
import unicodedata
import sys
import numpy as np
import pickle
import base64
from PIL import Image
import os, base64, mimetypes
import streamlit as st

# Tải dữ liệu NLTK cần thiết
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Khởi tạo các thành phần NLP
stop_w = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
punctuation = dict.fromkeys((i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith("P")), " ")

# Định nghĩa các hàm tiền xử lý văn bản
def remove_url(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http?://\S+\#|www\.\S+\#", " ", text)
    return re.sub(r"#(URL|url)_.*#", " ", text)

def remove_component_html(text: str):
    if not isinstance(text, str):
        return ""
    clean_text = re.sub(r"&[a-zA-Z0-9#]+;", "", text)
    soup = BeautifulSoup(clean_text, features="html.parser")
    return soup.text

def remove_non_english_letters(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\W\d_]', lambda m: '' if not re.match(r'[a-zA-Z]', m.group()) else m.group(), text)

def nomalize_space(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def split_and_segment_words(text):
    if not isinstance(text, str):
        return ""
    # Tách camel case (chữ hoa)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    # Tách từ dính bằng wordninja
    tokens = text.split()
    segmented = []
    for token in tokens:
        segmented.extend(wordninja.split(token))
    
    return ' '.join(segmented)

def nlp_process(text):
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = text.replace(".", " ")
    text = remove_url(text)
    text = remove_component_html(text)
    text = remove_non_english_letters(text)
    text = split_and_segment_words(text)
    
    text = text.translate(punctuation)
    text = nomalize_space(text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_w]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Tải mô hình
@st.cache_resource
def load_model():
    try:
        model = joblib.load('C:\\Users\\DELL 15\\Documents\\Zalo Received Files\\fraud_detection_model_smote.pkl')
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        return None

model = load_model() 
WINDOWS_LOGO_PATH = r"C:\Users\DELL 15\Pictures\da-hoc-khoa-hoc-hue-3png-1.webp"  # <-- chỉnh đúng file

def _logo_src(path_asset="assets/icon/logo.png", path_win=WINDOWS_LOGO_PATH):
    chosen = path_win if (path_win and os.path.exists(path_win)) else path_asset
    if not os.path.exists(chosen):
        return None
    mime = mimetypes.guess_type(chosen)[0] or "image/png"
    with open(chosen, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def show_logo(top_px=100, right_px=24, width=70):
    src = _logo_src()
    if not src:
        return
    html = """
    <style>
      .logo-fixed { position: fixed; top: %dpx; right: %dpx; z-index: 1000; }
    </style>
    <div class="logo-fixed">
      <img src="%s" width="%d"/>
    </div>
    """ % (top_px, right_px, src, width)
    st.markdown(html, unsafe_allow_html=True)

show_logo()
# ==== END Floating Logo ====

with st.expander("📘 Hướng dẫn sử dụng ứng dụng"):
    st.markdown("""
    - Nhập **Tiêu đề** và **Mô tả công việc** là bắt buộc.
    - Các trường khác giúp mô hình hoạt động chính xác hơn.
    - Ứng dụng sử dụng mô hình học máy để dự đoán mức độ đáng tin cậy của tin tuyển dụng.
    """)
# Giao diện Streamlit
st.title("🕵️ Fake Job Posting Detector")
st.markdown("Nhập thông tin công việc để kiểm tra khả năng là tin tuyển dụng giả")

with st.form("job_form"):
    st.header("Thông tin công việc")
    
    col1, col2 = st.columns(2)
    # Cột 1
    title = col1.text_input("Tiêu đề*", help="Tiêu đề công việc")
    location = col1.text_input("Địa điểm", help="Địa điểm làm việc")
    department = col1.text_input("Phòng ban", help="Phòng ban tuyển dụng")
    salary_range = col1.text_input("Mức lương", help="Ví dụ: $50,000 - $70,000")
    
    # Cột 2
    telecommuting = col2.checkbox("Làm việc từ xa", help="Có thể làm việc từ xa")
    has_company_logo = col2.checkbox("Có logo công ty", help="Tin có logo công ty")
    has_questions = col2.checkbox("Có câu hỏi ứng tuyển", help="Yêu cầu trả lời câu hỏi khi ứng tuyển")
    
    # Trường văn bản dài
    company_profile = st.text_area("Thông tin công ty", height=100)
    description = st.text_area("Mô tả công việc*", height=150)
    requirements = st.text_area("Yêu cầu công việc", height=150)
    benefits = st.text_area("Phúc lợi", height=100)
    
    # Trường phân loại
    exp_options = ["Not Applicable", "Associate", "Director", "Entry level", 
                  "Executive", "Internship", "Mid-Senior level"]
    
    edu_options = ["Unspecified", "Bachelor's Degree", "Master's Degree", 
                  "High School or equivalent", "Some College Coursework Completed",
                  "Vocational", "Certification", "Associate Degree", "Professional"]
    
    type_options = ["Other","Full-time", "Part-time", "Contract", "Temporary", "Internship"]
    
    col3, col4, col5 = st.columns(3)
    employment_type = col3.selectbox("Loại hình công việc", options=type_options)
    required_experience = col4.selectbox("Yêu cầu kinh nghiệm", options=exp_options)
    required_education = col5.selectbox("Yêu cầu học vấn", options=edu_options)
    
    col6, col7 = st.columns(2)
    industry = col6.text_input("Ngành nghề", help="Ví dụ: Công nghệ thông tin")
    function = col7.text_input("Chức năng công việc", help="Loại công việc")
    
    # Nộp form
    submitted = st.form_submit_button("Kiểm tra công việc")
    
    if submitted:
        # Kiểm tra trường bắt buộc
        if not title or not description:
            st.error("Vui lòng nhập Tiêu đề và Mô tả công việc")
        else:
            # Tạo dictionary dữ liệu
            job_data = {
                "title": title,
                "location": location,
                "department": department,
                "salary_range": salary_range,
                "company_profile": company_profile,
                "description": description,
                "requirements": requirements,
                "benefits": benefits,
                "telecommuting": int(telecommuting),
                "has_company_logo": int(has_company_logo),
                "has_questions": int(has_questions),
                "employment_type": employment_type,
                "required_experience": required_experience,
                "required_education": required_education,
                "industry": industry,
                "function": function
            }
            
            # Tạo cột missing
            job_data["missing_employment_type"] = int(not employment_type)
            job_data["missing_benefits"] = int(not benefits)
            job_data["missing_description"] = int(not description)
            job_data["missing_company_profile"] = int(not company_profile)
            job_data["missing_location"] = int(not location)
            job_data["missing_department"] = int(not department)
            
            # Tạo cột text
            text_fields = [title, location, department, description, 
                          company_profile, benefits, requirements, salary_range]
            job_data["text"] = " ".join(filter(None, text_fields))
            
            # Tiền xử lý văn bản
            job_data["text"] = job_data["text"].lower()
            job_data["text"] = remove_url(job_data["text"])
            job_data["text"] = remove_component_html(job_data["text"])
            job_data["text"] = remove_non_english_letters(job_data["text"])
            job_data["text"] = split_and_segment_words(job_data["text"])
            job_data["text"] = job_data["text"].translate(punctuation)
            job_data["text"] = nomalize_space(job_data["text"])
            
            # Tạo DataFrame
            df = pd.DataFrame([job_data])
            
            # Sắp xếp cột theo đúng thứ tự mô hình yêu cầu
            columns_order = [
                'telecommuting', 'has_company_logo', 'has_questions',
                'missing_employment_type', 'missing_benefits', 'missing_description',
                'missing_company_profile', 'missing_location', 'missing_department',
                'text', 'employment_type', 'required_experience',
                'required_education', 'industry', 'function'
            ]
            df = df[columns_order]
            
            # Dự đoán
            try:
                prediction = model.predict(df)
                prediction_proba = model.predict_proba(df)
                
                # Hiển thị kết quả
                st.subheader("Kết quả kiểm tra")
                if prediction[0] == 1:
                    st.error("⚠️ CẢNH BÁO: Công việc có khả năng là TIN TUYỂN DỤNG GIẢ")
                else:
                    st.success("✅ Công việc có vẻ là TIN TUYỂN DỤNG THẬT")
                
                # Hiển thị xác suất
                proba = prediction_proba[0][0] * 100
                st.metric(label="Độ tin cậy dự đoán", 
                          value=f"{proba:.1f}%", 
                          help="Tỷ lệ phần trăm khả năng là công việc thật")
                #proba = prediction_proba[0][1] * 100
                    #st.metric(label="Độ tin cậy dự đoán", 
                                #value=f"{proba:.1f}%", 
                               # help="Tỷ lệ phần trăm khả năng là công việc giả")
                
                # Giải thích kết quả
                st.subheader("Phân tích kết quả")
                if prediction[0] == 1:
                    st.write("Mô hình của chúng tôi đã phát hiện các dấu hiệu đáng ngờ trong tin tuyển dụng này. "
                             "Dưới đây là một số yếu tố thường gặp trong tin tuyển dụng giả:")
                    st.markdown("- Thiếu thông tin chi tiết về công ty hoặc công việc")
                    st.markdown("- Mô tả công việc mơ hồ, không rõ ràng")
                    st.markdown("- Yêu cầu quá đơn giản so với mức lương hấp dẫn")
                    st.markdown("- Thông tin liên hệ không đáng tin cậy")
                else:
                    st.write("Tin tuyển dụng này có vẻ hợp lệ với các thông tin chi tiết và minh bạch. "
                             "Tuy nhiên, luôn cảnh giác với:")
                    st.markdown("- Yêu cầu thanh toán trước khi nhận việc")
                    st.markdown("- Công ty không có thông tin rõ ràng trên mạng")
                    st.markdown("- Quy trình tuyển dụng quá đơn giản")
                
                # Hiển thị dữ liệu đã nhập
                st.subheader("Dữ liệu đã nhập")
                st.dataframe(df.iloc[:, :10])  # Hiển thị 10 cột đầu
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra khi dự đoán: {str(e)}")

# Thông tin thêm
st.markdown("---")
st.subheader("Lưu ý quan trọng")
st.markdown("""
- Mô hình này chỉ cung cấp đánh giá dựa trên dữ liệu nhập vào
- Kết quả dự đoán không đảm bảo 100% chính xác
- Luôn kiểm tra kỹ thông tin công ty và quy trình tuyển dụng
- Không cung cấp thông tin cá nhân nhạy cảm trước khi xác minh
- Các trường có dấu * là bắt buộc
""")

st.caption("Ứng dụng phát hiện tin tuyển dụng giả | Phiên bản 1.0 | Sử dụng mô hình Machine Learning")
## Đóng gói file pkl của mô hình 
## vd pipeline.fit(X_train, y_train)

# Đánh giá mô hình
#y_pred = pipeline.predict(X_test)

#print("\nModel Evaluation:")
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# Lưu mô hình thành file .pkl
#dump(pipeline, 'fraud_detection_model.pkl')
#print("\nModel has been saved as 'fraud_detection_model.pkl'")
# Để load lại mô hình sau này, sử dụng:
# pipeline = load('fraud_detection_model.pkl')

## Sau đó doc file tai mo hinh vao de train 
# Tải mô hình
#@st.cache_resource
#def load_model():
 #   try:
  #      model = joblib.load('C:\\Users\\DELL 15\\Documents\\Zalo Received Files\\fraud_detection_model.pkl')
   #     st.sidebar.success(" Model loaded successfully!")
    #    return model
    #except Exception as e:
     #   st.sidebar.error(f" Model loading failed: {str(e)}")
      #  return None

#model = load_model()

#vấn đề gặp phải là phải chạy trên cmd mới ra dự án được cần biến nó thành link để demo

##  cd "C:\Users\DELL 15\Downloads"
## python -m streamlit run demo_streamlitgiaodien.py (code để chạy)