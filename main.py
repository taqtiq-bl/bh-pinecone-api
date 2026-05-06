from flask import Flask, request, jsonify
from pinecone import Pinecone
import os

app = Flask(__name__)

# Pinecone Client initialisieren
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(host=os.environ.get('PINECONE_HOST'))

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    
    suchbegriff = data.get('suchbegriff', '')
    segment = data.get('segment')
    region = data.get('region')
    top_k = data.get('limit', 100)
    
    # Filter aufbauen - nur wenn Werte vorhanden und nicht leer
    filter_dict = {}
    if segment and segment.strip():
        filter_dict['segment'] = segment.strip()
    if region and region.strip():
        filter_dict['region'] = region.strip()
    
    # Zuerst Embedding generieren via Pinecone Inference
    embedding_response = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[suchbegriff],
        parameters={"input_type": "query"}
    )
    
    query_vector = embedding_response.data[0].values
    
    # Dann mit Vektor suchen
    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    # Filter nur hinzufügen wenn nicht leer
    if filter_dict:
        query_params["filter"] = filter_dict
    
    results = index.query(**query_params)
    
    # Ergebnisse formatieren
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
