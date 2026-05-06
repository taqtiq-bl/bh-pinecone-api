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
    
    # Filter aufbauen
    filter_dict = {}
    if segment:
        filter_dict['segment'] = {'$eq': segment}
    if region:
        filter_dict['region'] = {'$eq': region}
    
    # Pinecone Suche mit integrated inference
    results = index.search(
        namespace='',
        query={
            'top_k': top_k,
            'inputs': {'text': suchbegriff},
            'filter': filter_dict if filter_dict else None
        },
        fields=['firma', 'kbo_nummer', 'segment', 'nace_codes', 'adresse', 'plz', 'stadt', 'region', 'sprache']
    )
    
    # Ergebnisse formatieren
    hits = []
    for hit in results.result.hits:
        hits.append({
            'score': hit.score,
            **hit.fields
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
