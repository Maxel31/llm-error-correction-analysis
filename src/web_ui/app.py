"""
Flask application for visualizing activation differences.
"""
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/list_results')
def list_results():
    """List available result files."""
    results_dir = Path(os.environ.get('RESULTS_DIR', str(PROJECT_ROOT / 'data' / 'results')))
    
    print(f"Looking for results in: {results_dir}")
    print(f"Results directory exists: {results_dir.exists()}")
    
    if not results_dir.exists():
        return jsonify({"error": "Results directory not found"}), 404
    
    result_files = []
    for file in results_dir.glob('**/*.json'):
        if file.is_file():
            print(f"Found file: {file}")
            result_files.append({
                "path": str(file),
                "name": file.stem,
                "full_path": str(file)
            })
    
    print(f"Total result files found: {len(result_files)}")
    return jsonify({"results": result_files})

@app.route('/load_result')
def load_result():
    """Load a specific result file."""
    file_path = request.args.get('path')
    
    if not file_path:
        return jsonify({"error": "No file path provided"}), 400
    
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        layers = []
        sentence_types = []
        
        if "diff_token_type" in data:
            for pair_type, subtypes in data["diff_token_type"].items():
                if pair_type not in sentence_types:
                    sentence_types.append(pair_type)
                
                for subtype, pairs in subtypes.items():
                    for pair_id, pair_data in pairs.items():
                        if "layers" in pair_data:
                            for layer_name in pair_data["layers"].keys():
                                if layer_name not in layers:
                                    layers.append(layer_name)
        
        return jsonify({
            "data": data,
            "metadata": {
                "layers": sorted(layers, key=lambda x: int(x.split("_")[1]) if "_" in x else 0),
                "sentence_types": sentence_types
            }
        })
    
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {file_path}"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": f"Invalid JSON file: {file_path}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error loading file: {str(e)}"}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
