# PanKgraph AI Assistant

## Setup

1. Install dependencies  
   Make sure you have Python 3.9+ installed. Then install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Configure environment variables  
   Copy the provided example environment file and update it with your own API keys and settings:
   ```
   cp .env.example .env
   ```
   Open .env in your editor and fill in the missing values (e.g., API keys).  

3. Add config.py
   ```
   touch config.py
   ```
   Next, add the following lines:
   ```
   API_KEY='<Your-API-Key>'
   OPENAI_API_KEY=API_KEY
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

## Development

You can review the functions and interfaces in `main.py` and modify them as needed. The main entry point supports both modes automatically based on command-line arguments.
