from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pytesseract')
import pytesseract
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello, World!'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        extracted_text = pytesseract.image_to_string(img)
        return jsonify({'extracted_text': extracted_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
