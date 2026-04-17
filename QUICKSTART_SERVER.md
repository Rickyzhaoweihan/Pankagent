# Quick Start: PlannerAgent API Server

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install server requirements
pip install -r requirements-server.txt
```

### 2. Start the Server

```bash
# Default port (8080)
python3 server.py

# Or specify a custom port
python3 server.py 9000

# Or use environment variable
PORT=5000 python3 server.py
```

You'll see:
```
Starting PlannerAgent API Server...
Pre-loading Text2Cypher agent...
✓ Agents initialized successfully in 6.23s
Server is ready to accept requests!

INFO:     Uvicorn running on http://0.0.0.0:8080
```

### 3. Make Your First Request

**Option A: Using curl**
```bash
# If using default port 8080
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which SNP serves as the lead QTL for CFTR?"}'

# Or replace 8080 with your chosen port
```

**Option B: Using Python**
```python
import requests

# Replace 8080 with your chosen port
response = requests.post(
    'http://localhost:8080/query',
    json={'question': 'What is gene TP53?'}
)

print(response.json()['answer'])
```

**Option C: Using the interactive API docs**
- Open your browser to: http://localhost:8080/docs (or your chosen port)
- Click "Try it out" on the `/query` endpoint
- Enter your question and click "Execute"

---

## 📊 Available Endpoints

### `GET /` - API Information
```bash
curl http://localhost:8000/
```

### `GET /health` - Health Check
```bash
curl http://localhost:8000/health
```

### `POST /query` - Submit a Question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

### `GET /docs` - Interactive API Documentation
Open in browser: http://localhost:8000/docs

---

## 🧪 Test the Server

Run the test suite:
```bash
# Test default port (8080)
python3 test_server.py

# Test custom port
python3 test_server.py 9000

# Or with environment variable
PORT=5000 python3 test_server.py
```

This will test:
- ✓ Health endpoint
- ✓ Root endpoint
- ✓ Query processing
- ✓ Error handling

---

## 🎯 Example Queries

```bash
# Gene information
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the function of gene CFTR?"}'

# Disease associations
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What genes are associated with type 1 diabetes?"}'

# SNP relationships
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Find SNPs that have QTL relationships with gene MAFA"}'

# Cell type expression
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which genes are upregulated in Alpha Cell?"}'
```

---

## ⚡ Performance

### First Request
- **Initialization time:** ~6 seconds (happens once at server startup)
- **Query processing:** ~2-5 seconds

### Subsequent Requests
- **Query processing:** ~2-5 seconds (no initialization overhead!)

### Why Use the Server?
- ✅ **Agents load once** - No 6-second delay per query
- ✅ **Cache persists** - All models stay in memory
- ✅ **Multiple users** - Handle concurrent requests
- ✅ **Production ready** - Easy to deploy on AWS

---

## 🔧 Configuration

### Change Port
```bash
# Method 1: Command line argument
python3 server.py 9000

# Method 2: Environment variable
PORT=5000 python3 server.py

# Method 3: Set permanently in your shell
export PORT=8080
python3 server.py
```

### Production Mode
```bash
# Run with multiple workers
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4

# Or with gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 🌐 Deploy to AWS

See [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for detailed deployment instructions:
- **EC2:** Simple, full control (~$30-50/month)
- **ECS Fargate:** Production-ready, auto-scaling (~$100-200/month)
- **Lambda:** Serverless, pay-per-use (~$0-50/month)

**Recommended:** ECS Fargate for production

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process
kill -9 <PID>
```

### Import errors
```bash
# Make sure you're in the right directory
cd /path/to/PlannerAgent_3_11-2-2025

# Reinstall dependencies
pip install -r requirements.txt
pip install -r requirements-server.txt
```

### Slow responses
- First request is always slower (agent initialization)
- Check your API keys are configured correctly
- Monitor logs for errors: `tail -f logs/performance.log`

---

## 📚 Next Steps

1. ✅ Start the server locally
2. ✅ Test with example queries
3. ✅ Integrate into your application
4. ✅ Deploy to AWS for production use

For more details:
- [USAGE.md](USAGE.md) - Detailed usage examples
- [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) - Production deployment guide
- [README.md](README.md) - Full documentation

