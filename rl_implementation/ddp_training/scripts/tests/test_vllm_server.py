#!/usr/bin/env python3
"""
Test script to verify vLLM OpenAI-compatible server is working correctly.

Usage:
    python test_vllm_server.py --port 8001
    python test_vllm_server.py --port 8001 --prompt "Hello, world!"
"""

import argparse
import json
import requests
import sys


def test_health(host: str, port: int) -> bool:
    """Test the health endpoint."""
    url = f"http://{host}:{port}/health"
    print(f"\n1. Testing health endpoint: {url}")
    try:
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200] if response.text else '(empty)'}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_models(host: str, port: int) -> str:
    """Test the models endpoint to get available model names."""
    url = f"http://{host}:{port}/v1/models"
    print(f"\n2. Testing models endpoint: {url}")
    try:
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)[:500]}")
            if data.get("data"):
                model_id = data["data"][0]["id"]
                print(f"   Model ID: {model_id}")
                return model_id
        else:
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   Error: {e}")
    return None


def test_completions(host: str, port: int, model: str, prompt: str) -> bool:
    """Test the completions endpoint."""
    url = f"http://{host}:{port}/v1/completions"
    print(f"\n3. Testing completions endpoint: {url}")
    print(f"   Model: {model}")
    print(f"   Prompt: {prompt[:100]}...")
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    
    print(f"\n   Request payload:")
    print(f"   {json.dumps(payload, indent=2)[:500]}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"\n   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)[:1000]}")
            if data.get("choices"):
                text = data["choices"][0].get("text", "")
                print(f"\n   Generated text: {text[:200]}...")
            return True
        else:
            print(f"   Error response: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_chat_completions(host: str, port: int, model: str, prompt: str) -> bool:
    """Test the chat completions endpoint (alternative format)."""
    url = f"http://{host}:{port}/v1/chat/completions"
    print(f"\n4. Testing chat completions endpoint: {url}")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "temperature": 0.7,
    }
    
    print(f"   Request payload:")
    print(f"   {json.dumps(payload, indent=2)[:500]}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"\n   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)[:1000]}")
            return True
        else:
            print(f"   Error response: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"   Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test vLLM OpenAI-compatible server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--prompt", type=str, default="What is 2 + 2?", help="Test prompt")
    parser.add_argument("--model", type=str, default=None, help="Model name (auto-detected if not provided)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("vLLM Server Test")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    
    # Test 1: Health
    if not test_health(args.host, args.port):
        print("\n❌ Health check failed. Is the server running?")
        sys.exit(1)
    print("   ✓ Health check passed")
    
    # Test 2: Get models
    model = args.model or test_models(args.host, args.port)
    if not model:
        print("\n⚠️  Could not get model name. Using placeholder.")
        model = "unknown"
    
    # Test 3: Completions
    print("\n" + "=" * 60)
    print("Testing text generation...")
    print("=" * 60)
    
    if test_completions(args.host, args.port, model, args.prompt):
        print("\n   ✓ Completions endpoint works!")
    else:
        print("\n   ❌ Completions endpoint failed")
        
        # Try chat completions as fallback
        print("\n   Trying chat completions as alternative...")
        if test_chat_completions(args.host, args.port, model, args.prompt):
            print("\n   ✓ Chat completions endpoint works!")
        else:
            print("\n   ❌ Chat completions also failed")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


