from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)

# Configuration
FASTAPI_URL = "http://localhost:8000"
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'wav', 'mp3', 'ogg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app.route('/predict/text', methods=['POST'])
def predict_text():
    """Handle text-only prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Forward to FastAPI backend
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json={"text": text}
        )
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/audio', methods=['POST'])
def predict_audio():
    """Handle audio-only prediction"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporarily
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Forward to FastAPI backend
        with open(filepath, 'rb') as f:
            files = {'audio': (filename, f, audio_file.content_type)}
            response = requests.post(f"{FASTAPI_URL}/predict", files=files)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/video', methods=['POST'])
def predict_video():
    """Handle video-only prediction"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporarily
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Forward to FastAPI backend
        with open(filepath, 'rb') as f:
            files = {'video': (filename, f, video_file.content_type)}
            response = requests.post(f"{FASTAPI_URL}/predict", files=files)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/fusion', methods=['POST'])
def predict_fusion():
    """Handle fusion prediction (audio + video)"""
    try:
        files_to_send = {}
        temp_files = []
        
        # Handle video
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file.filename != '':
                filename = secure_filename(video_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                video_file.save(filepath)
                temp_files.append(filepath)
                
                with open(filepath, 'rb') as f:
                    files_to_send['video'] = (filename, f.read(), video_file.content_type)
        
        # Handle audio (if separate)
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename != '':
                filename = secure_filename(audio_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(filepath)
                temp_files.append(filepath)
                
                with open(filepath, 'rb') as f:
                    files_to_send['audio'] = (filename, f.read(), audio_file.content_type)
        
        if not files_to_send:
            return jsonify({'error': 'No files provided'}), 400
        
        # Forward to FastAPI backend
        response = requests.post(f"{FASTAPI_URL}/predict", files=files_to_send)
        
        # Clean up
        for filepath in temp_files:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        # Clean up on error
        for filepath in temp_files:
            if os.path.exists(filepath):
                os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask Web Interface...")
    print("📍 Open http://localhost:5000 in your browser")
    print("🔗 FastAPI backend should be running on http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=5000)
