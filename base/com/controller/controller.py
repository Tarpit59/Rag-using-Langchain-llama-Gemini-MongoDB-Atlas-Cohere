from flask import render_template, request, jsonify
from werkzeug.utils import secure_filename
from base.com.service.service import process_uploaded_pdfs, get_rag_response
from base import app
import os

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' in request.files:
        files = request.files.getlist('files')
        if files:
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

            process_uploaded_pdfs(app.config['UPLOAD_FOLDER'])

            return jsonify({"message": "Files uploaded and processed successfully"})
    
    return jsonify({"error": "No files provided"}), 400

@app.route('/query', methods=['POST'])
def query_rag():
    model_choice = request.form.get('model')
    question = request.form.get('question')

    try:
        response = get_rag_response(model_choice, question)
        return jsonify({"response": response})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400