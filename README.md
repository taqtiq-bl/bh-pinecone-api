# Pinecone Search API für B+H Lead Generation

## Deploy auf Railway

### 1. GitHub Repository erstellen
- Neues Repository auf GitHub erstellen (z.B. `bh-pinecone-api`)
- Diese 3 Dateien hochladen: `main.py`, `requirements.txt`, `Procfile`

### 2. Railway verbinden
- Geh zu https://railway.app/
- "New Project" → "Deploy from GitHub repo"
- Wähle dein Repository

### 3. Environment Variables setzen
In Railway → Variables:
```
PINECONE_API_KEY = <dein Pinecone API Key — nur in Railway Variables, nie ins Repo>
PINECONE_HOST = <dein Pinecone Index Host>
```

### 4. Deploy
Railway deployed automatisch. Du bekommst eine URL wie:
`https://bh-pinecone-api-production.up.railway.app`

## n8n Integration

HTTP Request Node mit:
- **Method:** POST
- **URL:** `https://DEINE-RAILWAY-URL/search`
- **Body (JSON):**
```json
{
  "suchbegriff": "{{ $json.Suchbegriff }}",
  "segment": "{{ $json.Segment }}",
  "region": "{{ $json.Region }}",
  "limit": 100
}
```

## Test
```bash
curl -X POST https://DEINE-RAILWAY-URL/search \
  -H "Content-Type: application/json" \
  -d '{"suchbegriff": "Lackfabrik", "segment": "Chemie", "limit": 10}'
```
