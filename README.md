# PanKgraph AI Assistant

Multi-agent system for querying biomedical knowledge graphs using natural language. Combines specialized agents (PankBase, GLKB, TemplateToolAgent) with a local fine-tuned LLM for text-to-Cypher translation.

## Prerequisites

- Python 3.9+
- GPU with CUDA support (for local model inference)
- 16GB+ GPU memory recommended (for vLLM)

## Setup

### 1. Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install server requirements (for API mode)
pip install -r requirements-server.txt

# Install vLLM (for local model inference)
pip install vllm
```

### 2. Configure API Keys

Create `config.py` with your API keys:

```bash
touch config.py
```

Add the following (replace with your actual keys):

```python
API_KEY='your-anthropic-api-key-here'
OPENAI_API_KEY='your-openai-api-key-here'
```

⚠️ **Security Note:** `config.py` is in `.gitignore` and will NOT be committed to git.

### 3. Start vLLM Server (Local Model)

The system uses a fine-tuned model for text-to-Cypher translation. Start the vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/neo4j_merged_model \
  --served-model-name text2cypher_merged \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

**For multi-user support, add:**
```bash
  --max-num-seqs 32  # Allows 32 concurrent requests
```

**Example with full path:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /scratch/drjieliu_root/drjieliu/rickyhan/neo4j_merged_model \
  --served-model-name text2cypher_merged \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 32
```

The vLLM server will be available at `http://localhost:8000/v1`

**Verify vLLM is running:**
```bash
curl http://localhost:8000/v1/models
```

## Usage

### Option 1: API Server (Recommended for Production)

Run as a web server for fast, persistent responses:

```bash
# Install server dependencies
pip install -r requirements-server.txt

# Start the server (default port 8080)
python3 server.py

# Or specify a custom port
python3 server.py 9000

# Or use environment variable
PORT=5000 python3 server.py
```

The server will:
- Initialize all agents once at startup (takes ~6 seconds)
- Stay running and respond to HTTP requests instantly
- Be available at `http://localhost:8080` (or your chosen port)
- Provide API documentation at `http://localhost:8080/docs`

**Make requests:**
```bash
# Using curl (replace 8080 with your port)
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gene TP53?"}'

# Using Python
import requests
response = requests.post('http://localhost:8080/query', 
                        json={'question': 'What is gene TP53?'})
print(response.json()['answer'])
```

See [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for production deployment on AWS.

### Option 2: Single Round Mode (Command Line)

Ask a single question and get an answer (program exits after responding):

```bash
python3 main.py "What is the function of gene TP53?"
```

The program will:
1. Process your question
2. Query the knowledge graph and relevant agents
3. Print the final formatted answer
4. Exit

### Option 3: Interactive Mode (Development)

Run without arguments for continuous conversation:

```bash
python3 main.py
```

This starts an interactive session where you can ask multiple questions in sequence.

### More Examples

See [USAGE.md](USAGE.md) for detailed usage examples and integration patterns.

## Architecture

The system uses a multi-agent architecture:

- **PlannerAgent** (`main.py`) - Orchestrates calls to specialized agents
- **PankBaseAgent** - Queries the knowledge graph using text-to-Cypher translation
- **GLKBAgent** - Searches biomedical literature abstracts
- **TemplateToolAgent** - Matches queries to predefined templates
- **FormatAgent** - Formats and combines results from all agents

### Text-to-Cypher Pipeline

1. Natural language query → PankBaseAgent
2. vLLM (fine-tuned model) → Generates Cypher query
3. Cypher validation & refinement
4. Execute against Neo4j knowledge graph
5. Return structured results

## Development

You can review the functions and interfaces in `main.py` and modify them as needed. The main entry point supports both modes automatically based on command-line arguments.

## 🔒 Security

**Never commit these files to git:**
- `config.py` - Contains API keys
- `.env` - Environment variables
- Any files with credentials or secrets

These are already in `.gitignore`. Use `.env.example` as a template.

## 📚 Documentation

- [USAGE.md](USAGE.md) - Detailed usage examples
- [QUICKSTART_SERVER.md](QUICKSTART_SERVER.md) - Quick start guide for API server
- [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) - Production deployment on AWS
- [PORT_CONFIGURATION.md](PORT_CONFIGURATION.md) - Port configuration guide
- [AGENTS.md](AGENTS.md) - Agent architecture details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

See [LICENSE](LICENSE) file for details.
