from flask import Flask, render_template, request, jsonify
import os
import time
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Simulate progress for saving
        # In a real app, you might stream the upload and report progress
        return jsonify({'filepath': filepath})
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/build_index', methods=['POST'])
def build_index():
    data = request.get_json()
    filepath = data.get('filepath')
    sparsity = data.get('sparsity')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400

    # Simulate index building progress
    # In a real app, this would be a long-running task
    # You might use Celery or a similar task queue to handle this
    
    # Placeholder for index size
    index_size = os.path.getsize(filepath) * (1 - float(sparsity)) / 1024  # Simulate size in KB

    return jsonify({
        'message': 'Index built successfully',
        'index_size': f'{index_size:.2f} KB'
    })

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_str = data.get('query')
    use_index = data.get('use_index', False)
    
    # Simulate inference time
    start_time = time.time()
    time.sleep(2) # Simulate work
    end_time = time.time()
    inference_time = end_time - start_time

    # Simulate results
    results = [
        {'id': 1, 'text': 'This movie was fantastic!', 'sentiment': 'positive', 'audience': 'all'},
        {'id': 3, 'text': 'A truly heartwarming story.', 'sentiment': 'positive', 'audience': 'children'},
    ]

    # Simulate trace data
    trace_data = {
        "traceEvents": [
            {"ph": "X", "name": "KV Cache Transfer", "ts": 0, "dur": 50000, "pid": 1, "tid": 1, "args": {}},
            {"ph": "X", "name": "Computation", "ts": 25000, "dur": 100000, "pid": 1, "tid": 2, "args": {}},
            {"ph": "X", "name": "KV Cache Transfer", "ts": 110000, "dur": 60000, "pid": 1, "tid": 1, "args": {}},
            {"ph": "X", "name": "Computation", "ts": 130000, "dur": 120000, "pid": 1, "tid": 2, "args": {}}
        ]
    }

    return jsonify({
        'results': results,
        'inference_time': f'{inference_time:.4f} seconds',
        'trace_data': trace_data
    })

if __name__ == '__main__':
    app.run(debug=True, port=2025)
