from flask import Flask, request, jsonify, send_file, send_from_directory
from pinecone import Pinecone
import os
import uuid
import io
 
app = Flask(__name__)
 
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(host=os.environ.get('PINECONE_HOST'))
 
pdf_storage = {}
 
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
 
 
PROVINZ_ALIASES = {
    "bruxelles capitale": "Bruxelles",
    "bruxelles-capitale": "Bruxelles",
    "brussel": "Bruxelles",
    "brussels": "Bruxelles",
    "bruxelles": "Bruxelles",
    "liege": "Liège",
}
 
 
def resolve_provinz(provinz):
    """Provinz-Namen tolerant auflösen (Akzente, Aliase wie 'Bruxelles Capitale')."""
    if not provinz:
        return None
    raw = str(provinz).strip()
    if raw in PROVINZ_PLZ_RANGES:
        return raw
    import unicodedata
    key = unicodedata.normalize('NFD', raw.lower())
    key = ''.join(c for c in key if unicodedata.category(c) != 'Mn').strip()
    if key in PROVINZ_ALIASES:
        return PROVINZ_ALIASES[key]
    for name in PROVINZ_PLZ_RANGES:
        norm = unicodedata.normalize('NFD', name.lower())
        norm = ''.join(c for c in norm if unicodedata.category(c) != 'Mn')
        if norm == key:
            return name
    return None
 
 
def normalize_nace(codes):
    """NACE-Codes auf das Metadata-Format normalisieren: 5-stelliger String ohne Punkte."""
    arr = codes if isinstance(codes, list) else [codes]
    out = []
    for c in arr:
        if c is None or c == '':
            continue
        s = str(c).replace('.', '').replace(' ', '').strip()
        if s.isdigit():
            out.append(s)
    return out
 
 
def build_plz_filter(provinz=None, plz_von=None, plz_bis=None):
    provinz = resolve_provinz(provinz)
    if provinz and provinz in PROVINZ_PLZ_RANGES:
        ranges = PROVINZ_PLZ_RANGES[provinz]
        if len(ranges) == 1:
            von, bis = ranges[0]
            plz_list = [str(i) for i in range(int(von), int(bis) + 1)]
            return {"plz": {"$in": plz_list}}
        or_filters = []
        for von, bis in ranges:
            plz_list = [str(i) for i in range(int(von), int(bis) + 1)]
            or_filters.append({"plz": {"$in": plz_list}})
        return {"$or": or_filters}
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
    provinz = data.get('provinz')
    nace_codes = data.get('nace_codes', [])
    plz_von = data.get('plz_von')
    plz_bis = data.get('plz_bis')
    top_k = min(int(data.get('limit', 50)), 10000)
    min_score = float(data.get('min_score', 0.0))
 
    nace_arr = normalize_nace(nace_codes)
    deterministic = bool(nace_arr)
 
    if not suchbegriff and not deterministic:
        return jsonify({'error': 'suchbegriff is required (or provide nace_codes)'}), 400
 
    sub_filters = []
    if segment and segment.strip() and not deterministic:
        # Bei exakter NACE-Suche ist der Code die einzige Wahrheit --
        # ein zusaetzlicher Segment-Filter koennte korrekte Treffer ausschliessen.
        sub_filters.append({'segment': segment.strip()})
    if region and region.strip():
        sub_filters.append({'region': region.strip()})
    if nace_arr:
        # nace_list ist ein Array einzelner Codes (seit Metadata-Fix) -- $in matcht pro Element.
        # Fallback nace_codes deckt Alt-Records mit genau einem Code (exakter String-Match).
        sub_filters.append({'$or': [
            {'nace_list': {'$in': nace_arr}},
            {'nace_codes': {'$in': nace_arr}},
        ]})
    plz_filter = build_plz_filter(provinz=provinz, plz_von=plz_von, plz_bis=plz_bis)
    if plz_filter:
        sub_filters.append(plz_filter)
    
    if not sub_filters:
        filter_dict = None
    elif len(sub_filters) == 1:
        filter_dict = sub_filters[0]
    else:
        filter_dict = {'$and': sub_filters}
    
    if suchbegriff:
        embedding_response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[suchbegriff],
            parameters={"input_type": "query"}
        )
        query_vector = embedding_response.data[0].values
    else:
        # Reine NACE-Suche ohne Suchbegriff: Dummy-Vektor, der Metadata-Filter
        # bestimmt die Treffermenge vollstaendig.
        query_vector = [0.0] * 1024
 
    query_params = {"vector": query_vector, "top_k": top_k, "include_metadata": True}
    if filter_dict:
        query_params["filter"] = filter_dict
 
    results = index.query(**query_params)
 
    if deterministic:
        # Exakte NACE-Suche: min_score darf keine korrekten Treffer wegfiltern.
        min_score = 0.0
 
    hits = []
    for match in results.matches:
        score = float(match.score)
        if score < min_score:
            continue
        hits.append({'score': score, **match.metadata})
 
    if deterministic and not suchbegriff:
        # Ohne semantisches Ranking: stabile, reproduzierbare Reihenfolge.
        hits.sort(key=lambda h: (str(h.get('plz', '')), str(h.get('firma', '')).lower()))
 
    return jsonify({
        'count': len(hits),
        'total_from_pinecone': len(results.matches),
        'mode': 'deterministic' if deterministic else 'semantic',
        'nace_filter': nace_arr or None,
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
 
 
@app.route('/debug-nace', methods=['GET'])
def debug_nace():
    """
    Inventur aller NACE-Codes in der Pinecone-DB.
    Optionale Query-Params:
      - prefix=20    -> nur Codes die mit '20' anfangen
      - prefix=412   -> nur Codes die mit '412' anfangen
      - segment=Chemie -> nur dieses Segment
      - limit=5000   -> wie viele Records sampeln (default 5000, max 10000)
    """
    prefix = request.args.get('prefix', '').strip()
    segment = request.args.get('segment', '').strip()
    limit = min(int(request.args.get('limit', 5000)), 10000)
    
    filter_dict = {}
    if segment:
        filter_dict['segment'] = segment
    if prefix:
        if len(prefix) >= 5:
            possible_codes = [prefix[:5]]
        else:
            remaining = 5 - len(prefix)
            possible_codes = [prefix + str(i).zfill(remaining) for i in range(10**remaining)]
        filter_dict['nace_codes'] = {'$in': possible_codes}
    
    results = index.query(
        vector=[0.0] * 1024,
        top_k=limit,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )
    
    from collections import Counter
    code_counter = Counter()
    sample_firmen = {}
    
    for m in results.matches:
        code = m.metadata.get('nace_codes', '')
        if not code:
            continue
        code = str(code).strip()
        if prefix and not code.startswith(prefix):
            continue
        code_counter[code] += 1
        if code not in sample_firmen:
            sample_firmen[code] = {
                'firma': m.metadata.get('firma', ''),
                'stadt': m.metadata.get('stadt', ''),
                'plz': m.metadata.get('plz', ''),
                'segment': m.metadata.get('segment', '')
            }
    
    sorted_codes = [
        {
            'nace_code': code,
            'count': count,
            'beispiel_firma': sample_firmen[code]['firma'],
            'beispiel_stadt': sample_firmen[code]['stadt'],
            'beispiel_plz': sample_firmen[code]['plz'],
            'segment': sample_firmen[code]['segment']
        }
        for code, count in code_counter.most_common()
    ]
    
    return jsonify({
        'total_records_sampled': len(results.matches),
        'unique_nace_codes': len(code_counter),
        'filter_applied': filter_dict,
        'codes': sorted_codes
    })
 
 
@app.route('/logo.jpg')
def logo():
    return send_from_directory('/app', 'Logo_b+h_Claim_flaeche_farbe.jpg')
