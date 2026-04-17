# AWS Deployment Guide for PlannerAgent API

## Why FastAPI is Better for AWS

FastAPI is recommended for AWS deployment because:
- ✅ **Async support** - Better performance with concurrent requests
- ✅ **Automatic API docs** - Built-in Swagger UI at `/docs`
- ✅ **Type validation** - Pydantic models prevent errors
- ✅ **AWS Lambda support** - Can deploy as serverless with Mangum
- ✅ **ECS/EKS ready** - Easy containerization
- ✅ **API Gateway integration** - Works seamlessly with AWS services

## Deployment Options

### Option 1: AWS EC2 (Simplest, Full Control)

**Best for:** Development, testing, full control over environment

1. **Launch EC2 Instance:**
   ```bash
   # Choose: Ubuntu 22.04 LTS, t3.medium or larger
   # Open ports: 22 (SSH), 8000 (API), 80 (HTTP), 443 (HTTPS)
   ```

2. **Setup on EC2:**
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Install dependencies
   sudo apt update
   sudo apt install python3-pip python3-venv nginx -y
   
   # Clone/upload your code
   git clone your-repo.git
   cd PlannerAgent_3_11-2-2025
   
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-server.txt
   
   # Configure environment
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run with systemd (auto-restart):**
   ```bash
   # Create service file
   sudo nano /etc/systemd/system/planner-agent.service
   ```
   
   ```ini
   [Unit]
   Description=PlannerAgent API Server
   After=network.target
   
   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/PlannerAgent_3_11-2-2025
   Environment="PATH=/home/ubuntu/PlannerAgent_3_11-2-2025/.venv/bin"
   ExecStart=/home/ubuntu/PlannerAgent_3_11-2-2025/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   ```bash
   # Enable and start service
   sudo systemctl enable planner-agent
   sudo systemctl start planner-agent
   sudo systemctl status planner-agent
   ```

4. **Setup Nginx reverse proxy (optional):**
   ```nginx
   # /etc/nginx/sites-available/planner-agent
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

**Cost:** ~$30-50/month for t3.medium

---

### Option 2: AWS ECS with Fargate (Recommended for Production)

**Best for:** Production, scalability, managed infrastructure

1. **Create Dockerfile:**
   ```dockerfile
   # Dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements
   COPY requirements.txt requirements-server.txt ./
   
   # Install Python dependencies
   RUN pip install --no-cache-dir -r requirements.txt && \
       pip install --no-cache-dir -r requirements-server.txt
   
   # Copy application code
   COPY . .
   
   # Expose port
   EXPOSE 8000
   
   # Run the server
   CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
   ```

2. **Build and push to ECR:**
   ```bash
   # Create ECR repository
   aws ecr create-repository --repository-name planner-agent
   
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
   
   # Build image
   docker build -t planner-agent .
   
   # Tag and push
   docker tag planner-agent:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/planner-agent:latest
   docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/planner-agent:latest
   ```

3. **Create ECS Task Definition:**
   ```json
   {
     "family": "planner-agent",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "2048",
     "memory": "4096",
     "containerDefinitions": [
       {
         "name": "planner-agent",
         "image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/planner-agent:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {"name": "API_KEY", "value": "your-api-key"},
           {"name": "OPENAI_API_KEY", "value": "your-openai-key"}
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/planner-agent",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

4. **Create ECS Service with ALB:**
   - Create Application Load Balancer
   - Create Target Group (port 8000)
   - Create ECS Cluster
   - Create ECS Service with task definition
   - Configure auto-scaling (min: 2, max: 10)

**Cost:** ~$100-200/month (scales with usage)

---

### Option 3: AWS Lambda + API Gateway (Serverless)

**Best for:** Sporadic usage, cost optimization, auto-scaling

⚠️ **Note:** Lambda has cold start issues and 15-minute timeout. May not be ideal for long-running queries.

1. **Install Mangum (Lambda adapter):**
   ```bash
   pip install mangum
   ```

2. **Modify server.py:**
   ```python
   # Add at the end of server.py
   from mangum import Mangum
   handler = Mangum(app)
   ```

3. **Deploy with AWS SAM or Serverless Framework:**
   ```yaml
   # serverless.yml
   service: planner-agent
   
   provider:
     name: aws
     runtime: python3.11
     region: us-east-1
     timeout: 900  # 15 minutes max
     memorySize: 3008
     
   functions:
     api:
       handler: server.handler
       events:
         - http:
             path: /{proxy+}
             method: ANY
   ```

**Cost:** Pay per request, ~$0.20 per 1M requests

---

## Recommended Setup for Production

**Best Choice: ECS Fargate + ALB**

```
Internet → Route 53 → ALB → ECS Fargate (2-10 tasks) → RDS/ElastiCache (if needed)
```

**Why:**
- ✅ Managed infrastructure (no server management)
- ✅ Auto-scaling based on CPU/memory
- ✅ Load balancing across multiple instances
- ✅ Health checks and auto-recovery
- ✅ Easy CI/CD with CodePipeline
- ✅ Integrated with CloudWatch for monitoring

---

## Testing the Deployed API

```bash
# Health check
curl https://your-api.com/health

# Query
curl -X POST https://your-api.com/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gene TP53?"}'

# API Documentation
# Visit: https://your-api.com/docs
```

---

## Monitoring & Logging

1. **CloudWatch Logs:** All logs automatically sent
2. **CloudWatch Metrics:** CPU, memory, request count
3. **CloudWatch Alarms:** Alert on errors or high latency
4. **X-Ray:** Distributed tracing (optional)

---

## Security Best Practices

1. **Use Secrets Manager** for API keys
2. **Enable VPC** for ECS tasks
3. **Use Security Groups** to restrict access
4. **Enable HTTPS** with ACM certificates
5. **Add API authentication** (API keys, JWT, etc.)
6. **Rate limiting** with API Gateway or WAF

---

## Cost Comparison

| Option | Monthly Cost | Best For |
|--------|-------------|----------|
| EC2 t3.medium | $30-50 | Development, testing |
| ECS Fargate | $100-200 | Production, scaling |
| Lambda | $0-50 | Sporadic usage |

**Recommendation:** Start with **EC2** for development, move to **ECS Fargate** for production.

