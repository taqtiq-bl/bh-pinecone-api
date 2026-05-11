from flask import Flask, request, jsonify, send_file
from pinecone import Pinecone
import os
import uuid
import io

app = Flask(__name__)

# Pinecone Client initialisieren
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(host=os.environ.get('PINECONE_HOST'))

# In-memory storage für PDFs (wird bei Restart geleert)
pdf_storage = {}

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_id = str(uuid.uuid4())
    pdf_storage[file_id] = file.read()
    
    base_url = os.environ.get('BASE_URL', request.host_url.rstrip('/'))
    url = f"{base_url}/download/{file_id}"
    
    return jsonify({'url': url, 'id': file_id})

@app.route('/download/<file_id>', methods=['GET'])
def download(file_id):
    if file_id not in pdf_storage:
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        io.BytesIO(pdf_storage[file_id]),
        mimetype='application/pdf',
        as_attachment=False,
        download_name=f'{file_id}.pdf'
    )

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    
    suchbegriff = data.get('suchbegriff', '')
    segment = data.get('segment')
    region = data.get('region')
    top_k = data.get('limit', 100)
    
    filter_dict = {}
    if segment and segment.strip():
        filter_dict['segment'] = segment.strip()
    if region and region.strip():
        filter_dict['region'] = region.strip()
    
    embedding_response = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[suchbegriff],
        parameters={"input_type": "query"}
    )
    
    query_vector = embedding_response.data[0].values
    
    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    if filter_dict:
        query_params["filter"] = filter_dict
    
    results = index.query(**query_params)
    
    hits = []
    for match in results.matches:
        hits.append({
            'score': match.score,
            **match.metadata
        })
    
    return jsonify({
        'count': len(hits),
        'results': hits
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/logo.jpg')
def logo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, 'Logo_b+h_Claim_flaeche_farbe.jpg')
