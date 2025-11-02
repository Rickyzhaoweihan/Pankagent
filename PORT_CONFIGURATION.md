# Port Configuration Guide

The server now supports flexible port configuration! You can choose your port in three different ways.

## 🎯 Three Ways to Set the Port

### Method 1: Command Line Argument (Recommended)

```bash
# Start server on port 9000
python3 server.py 9000

# Start server on port 5000
python3 server.py 5000

# Test server on matching port
python3 test_server.py 9000
```

### Method 2: Environment Variable

```bash
# Set port via environment variable
PORT=9000 python3 server.py

# Or export it first
export PORT=9000
python3 server.py

# Test with same port
PORT=9000 python3 test_server.py
```

### Method 3: Default Port

```bash
# No arguments = uses default port 8080
python3 server.py

# Test default port
python3 test_server.py
```

## 📝 Examples

### Using Different Ports

```bash
# Development on port 8080 (default)
python3 server.py

# Production on port 80 (requires sudo)
sudo python3 server.py 80

# Alternative port if 8080 is busy
python3 server.py 8081

# Custom port for testing
python3 server.py 9999
```

### Testing the Server

```bash
# Start server on port 9000
python3 server.py 9000

# In another terminal, test it
python3 test_server.py 9000

# Or manually test
curl http://localhost:9000/health
```

## 🔍 Check Port Availability

### Find what's using a port:

```bash
# Linux/Mac
lsof -i :8080
netstat -tulpn | grep 8080

# Kill process using the port
kill -9 <PID>
```

### Common ports:

- `80` - HTTP (requires sudo)
- `443` - HTTPS (requires sudo)
- `8000` - Common development port
- `8080` - Alternative HTTP (default)
- `5000` - Flask default
- `3000` - Node.js default

## ⚙️ Production Configuration

### With Uvicorn directly:

```bash
# Single worker
uvicorn server:app --host 0.0.0.0 --port 8080

# Multiple workers
uvicorn server:app --host 0.0.0.0 --port 8080 --workers 4
```

### With Gunicorn:

```bash
gunicorn server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8080
```

### With systemd (auto-start on boot):

```ini
# /etc/systemd/system/planner-agent.service
[Service]
Environment="PORT=8080"
ExecStart=/path/to/.venv/bin/python server.py
```

## 🐳 Docker Configuration

```dockerfile
# Dockerfile
ENV PORT=8080
EXPOSE 8080
CMD ["python3", "server.py"]
```

```bash
# Run with custom port
docker run -p 9000:8080 planner-agent

# Or override the port
docker run -e PORT=9000 -p 9000:9000 planner-agent
```

## 🌐 AWS Deployment

### EC2:
```bash
# Open port in security group, then:
python3 server.py 8080
```

### ECS:
```json
{
  "portMappings": [
    {
      "containerPort": 8080,
      "hostPort": 8080
    }
  ]
}
```

### Lambda + API Gateway:
Port is managed by API Gateway, no configuration needed.

## ✅ Verification

After starting the server, verify it's running:

```bash
# Check health (replace 8080 with your port)
curl http://localhost:8080/health

# Expected response:
{
  "status": "healthy",
  "message": "Server is running and ready to accept requests",
  "uptime_seconds": 123.45
}
```

## 🚨 Troubleshooting

### Port already in use:
```bash
# Find and kill the process
lsof -i :8080
kill -9 <PID>

# Or use a different port
python3 server.py 8081
```

### Permission denied (ports < 1024):
```bash
# Use sudo for privileged ports
sudo python3 server.py 80

# Or use a higher port
python3 server.py 8080
```

### Can't connect from other machines:
```bash
# Server binds to 0.0.0.0 by default (all interfaces)
# Check firewall settings:
sudo ufw allow 8080
```

## 📚 Related Documentation

- [QUICKSTART_SERVER.md](QUICKSTART_SERVER.md) - Quick start guide
- [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) - AWS deployment
- [README.md](README.md) - Main documentation

