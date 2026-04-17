#!/bin/bash
# =============================================================================
# Stop all vLLM OpenAI-compatible Servers
# =============================================================================

echo "Stopping all vLLM servers..."

# Find and kill all vLLM server processes
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && echo "  Sent SIGTERM to vLLM servers" || echo "  No vLLM servers found"

# Wait a bit for graceful shutdown
sleep 2

# Force kill if still running
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null 2>&1; then
    echo "  Force stopping remaining servers..."
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 1
fi

# Verify
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null 2>&1; then
    echo "  Warning: Some servers may still be running"
    pgrep -af "vllm.entrypoints.openai.api_server"
else
    echo "  All vLLM servers stopped"
fi

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"

