from flask import Flask, request, jsonify, send_file, send_from_directory
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

# === Provinz -> PLZ-Range-Mapping (BE) ===
# Quelle: bpost / Statbel offizielle PLZ-Zuordnung
PROVINZ_PLZ_RANGES = {
    "Bruxelles":         [("1000", "1299")],
    "Brabant Wallon":    [("1300", "1499")],
    "Vlaams-Brabant":    [("1500", "1999"), ("3000", "3499")],
    "Antwerpen":         [("2000", "2999")],
    "Limburg":           [("3500", "3999")],
    "Liège":             [("4000", "4999")],
    "Namur":             [("5000", "5999")],
    "Hainaut":           [("6000", "6599"), ("7000", "7999")],
    "Luxembourg":        [("6600", "6999")],
    "West-Vlaanderen":   [("8000", "8999")],
    "Oost-Vlaanderen":   [("9000", "9999")],
}


def build_plz_filter(provinz=None, plz_von=None, plz_bis=None):
    """
    Baut Pinecone-Filter für PLZ. Drei Modi:
    1. Wenn `provinz` gegeben und in Mapping -> nutze Mapping (kann $or sein)
    2. Wenn `plz_von` + `plz_bis` gegeben -> nutze als einzelner Range
    3. Sonst kein PLZ-Filter
    """
    if provinz and provinz in PROVINZ_PLZ_RANGES:
        ranges = PROVINZ_PLZ_RANGES[provinz]
        
        if len(ranges) == 1:
            von, bis = ranges[0]
            plz_list = [str(i) for i in range(int(von), int(bis) + 1)]
            return {"plz": {"$in": plz_list}}
        
        # Mehrere Ranges -> $or-Konstruktion
        or_filters = []
        for von, bis in ranges:
            plz_list = [str(i) for i in range(int(von), int(bis) + 1)]
            or_filters.append({"plz": {"$in": plz_list}})
        return {"$or": or_filters}
    
    # Fallback: alter plz_von/plz_bis-Stil (Backwards-Compat)
    if plz_von and plz_bis:
        plz_list = [str(i) for i in range(int(plz_von), int(plz_bis) + 1)]
        return {"plz": {"$in": plz_list}}
    
    return None


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
    data = request.json or {}
    
    suchbegriff = data.get('suchbegriff', '')
    segment = data.get('segment')
    region = data.get('region')
    provinz = data.get('provinz')           # NEU
    nace_codes = data.get('nace_codes', []) # NEU
    plz_von = data.get('plz_von')           # Backwards-Compat
    plz_bis = data.get('plz_bis')           # Backwards-Compat
    top_k = int(data.get('limit', 50))
    min_score = float(data.get('min_score', 0.0))  # NEU
    
    if not suchbegriff:
        return jsonify({'error': 'suchbegriff is required'}), 400
    
    # === Sub-Filter sammeln ===
    sub_filters = []
    
    if segment and segment.strip():
        sub_filters.append({'segment': segment.strip()})
    
    if region and region.strip():
        sub_filters.append({'region': region.strip()})
    
    if nace_codes:
        # Auch single string akzeptieren
        nace_arr = nace_codes if isinstance(nace_codes, list) else [nace_codes]
        # Strings normalisieren (falls Punkte oder Whitespace drin)
        nace_arr = [str(c).replace('.', '').strip() for c in nace_arr if c]
        if nace_arr:
            sub_filters.append({'nace_codes': {'$in': nace_arr}})
    
    plz_filter = build_plz_filter(provinz=provinz, plz_von=plz_von, plz_bis=plz_bis)
    if plz_filter:
        sub_filters.append(plz_filter)
    
    # === Filter kombinieren ===
    if not sub_filters:
        filter_dict = None
    elif len(sub_filters) == 1:
        filter_dict = sub_filters[0]
    else:
        filter_dict = {'$and': sub_filters}
    
    # === Embedding für Suchbegriff erzeugen ===
    embedding_response = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[suchbegriff],
        parameters={"input_type": "query"}
    )
    query_vector = embedding_response.data[0].values
    
    # === Pinecone Query ===
    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }
    if filter_dict:
        query_params["filter"] = filter_dict
    
    results = index.query(**query_params)
    
    # === Score-Filter clientseitig ===
    hits = []
    for match in results.matches:
        score = float(match.score)
        if score < min_score:
            continue
        hits.append({
            'score': score,
            **match.metadata
        })
    
    return jsonify({
        'count': len(hits),
        'total_from_pinecone': len(results.matches),
        'filter_applied': filter_dict,
        'min_score_applied': min_score,
        'results': hits
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/debug')
def debug():
    import glob
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(base_dir + '/*')
    return jsonify({'base_dir': base_dir, 'files': files})


@app.route('/debug-plz', methods=['GET'])
def debug_plz():
    results = index.query(
        vector=[0.0] * 1024,
        top_k=5,
        include_metadata=True,
        filter={"region": "Flandre"}
    )
    plz_samples = [
        {
            "firma": m.metadata.get("firma"),
            "plz": m.metadata.get("plz"),
            "plz_type": type(m.metadata.get("plz")).__name__
        }
        for m in results.matches
    ]
    return jsonify({"samples": plz_samples})


@app.route('/logo.jpg')
def logo():
    return send_from_directory('/app', 'Logo_b+h_Claim_flaeche_farbe.jpg')
