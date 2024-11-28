from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from pyvi import ViTokenizer

app = Flask(__name__)
CORS(app)

# Đọc stopwords từ file
def load_stopwords(filepath='stopword.txt'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
        return stopwords
    except Exception as e:
        print(f"Lỗi khi đọc file stopwords: {str(e)}")
        return set()

# Load stopwords
STOPWORDS = load_stopwords()

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Chuyển text thành chữ thường và tách từ
    text = gensim.utils.simple_preprocess(text)
    text = ' '.join(text)
    
    # Tách từ tiếng Việt
    text = ViTokenizer.tokenize(text)
    
    # Loại bỏ stopwords
    words = text.split()
    words = [w for w in words if w.lower() not in STOPWORDS]
    text = ' '.join(words)
    
    return text

# Hàm chuyển đổi kết quả sang tiếng Việt có dấu
def format_prediction(prediction):
    category_mapping = {
        'Chinh tri Xa hoi': 'Chính trị Xã hội',
        'Doi song': 'Đời sống',
        'Khoa hoc': 'Khoa học',
        'Kinh doanh': 'Kinh doanh',
        'Phap luat': 'Pháp luật',
        'Suc khoe': 'Sức khỏe',
        'The gioi':'Thế giới',
        'The thao': 'Thể thao',
        'Van hoa': 'Văn hóa', 
        'Vi tinh': 'Vi tính'
    }
    return category_mapping.get(prediction, prediction)

# Tải model và vectorizer
try:
    tfidf_vect = joblib.load('tfidf_vectorizer.joblib')
    loaded_model = joblib.load('modelfinal.joblib')
except Exception as e:
    print(f"Lỗi khi tải model hoặc vectorizer: {str(e)}")

@app.route('/')
def home():
    return render_template('App.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Nhận request") # Debug log
        data = request.get_json(force=True)
        print("Data nhận được:", data) # Debug log
        
        if 'textInput' not in data:
            print("Không tìm thấy textInput") # Debug log
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy văn bản đầu vào'
            }), 400

        text = data['textInput']
        print("Văn bản gốc:", text) # Debug log
        
        if not text.strip():
            return jsonify({
                'success': False,
                'error': 'Văn bản không được để trống'
            }), 400

        # Tiền xử lý văn bản
        processed_text = preprocess_text(text)
        print("Văn bản sau xử lý:", processed_text) # Debug log
        
        if isinstance(processed_text, str):
            processed_text = [processed_text]
            
        # Chuyển văn bản thành vector
        text_vector = tfidf_vect.transform(processed_text)
        
        # Thực hiện dự đoán
        prediction = loaded_model.predict(text_vector)
        print("Kết quả dự đoán:", prediction[0]) # Debug log
        
        # Xử lý kết quả dự đoán
        prediction_result = prediction[0]
        
        # Chuyển đổi kết quả sang tiếng Việt có dấu
        formatted_result = format_prediction(prediction_result)
        print("Kết quả đã format:", formatted_result) # Debug log
        
        return jsonify({
            'success': True,
            'prediction': formatted_result,
            'message': 'Dự đoán thành công'
        })

    except Exception as e:
        print("Lỗi:", str(e)) # Debug log
        return jsonify({
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
