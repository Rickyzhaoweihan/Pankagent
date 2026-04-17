#!/usr/bin/env python3
"""
Diagnostic script to debug vLLM server issues.

This script tests each vLLM server to identify HTTP 500 errors.

Usage:
    python diagnose_vllm.py
    python diagnose_vllm.py --port 8001
"""

import argparse
import json
import requests
import sys
from typing import Optional, Tuple


def test_server(host: str, port: int, verbose: bool = True) -> Tuple[bool, str, Optional[str]]:
    """
    Test a vLLM server comprehensively.
    
    Returns:
        Tuple of (success, message, model_id)
    """
    base_url = f"http://{host}:{port}"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing server at {base_url}")
        print(f"{'='*60}")
    
    # Step 1: Health check
    if verbose:
        print("\n1. Health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code != 200:
            return False, f"Health check failed: HTTP {response.status_code}", None
        if verbose:
            print(f"   ✓ Health OK")
    except Exception as e:
        return False, f"Health check error: {e}", None
    
    # Step 2: Get model ID
    if verbose:
        print("\n2. Getting model ID from /v1/models...")
    model_id = None
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                model_id = data["data"][0]["id"]
                if verbose:
                    print(f"   ✓ Model ID: {model_id}")
            else:
                if verbose:
                    print(f"   ⚠ No models found in response: {data}")
        else:
            if verbose:
                print(f"   ⚠ Failed to get models: HTTP {response.status_code}")
    except Exception as e:
        if verbose:
            print(f"   ⚠ Error getting models: {e}")
    
    if not model_id:
        return False, "Could not determine model ID", None
    
    # Step 3: Simple completion test
    if verbose:
        print("\n3. Testing simple completion...")
    
    simple_prompt = "What is 2+2? Answer with just the number."
    
    try:
        payload = {
            "model": model_id,
            "prompt": simple_prompt,
            "max_tokens": 10,
            "temperature": 0.1,
        }
        
        if verbose:
            print(f"   Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("choices", [{}])[0].get("text", "")
            if verbose:
                print(f"   ✓ Response: {text[:100]}")
        else:
            error_text = response.text[:500]
            if verbose:
                print(f"   ✗ HTTP {response.status_code}")
                print(f"   Error: {error_text}")
            return False, f"Completion failed: HTTP {response.status_code}: {error_text}", model_id
            
    except Exception as e:
        return False, f"Completion error: {e}", model_id
    
    # Step 4: Test with longer prompt (similar to actual usage)
    if verbose:
        print("\n4. Testing with longer prompt...")
    
    long_prompt = """You are a question generator for a biomedical knowledge graph.

SCHEMA (core types):
Node Types:
  - gene: Represents genes with id, name, symbol
  - disease: Diseases with id, name, description
  - snp: Single nucleotide polymorphisms
  - cell_type: Cell types in pancreas

Edge Types:
  - associated_with: gene -> disease
  - regulates: gene -> gene
  - located_in: snp -> gene

CURRICULUM:
Difficulty: easy
Max hops: 1
Focus: simple single-hop queries

TASK: Generate ONE biomedical question that can be answered using the knowledge graph.

Generate question:"""

    try:
        payload = {
            "model": model_id,
            "prompt": long_prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }
        
        if verbose:
            print(f"   Prompt length: {len(long_prompt)} chars")
        
        response = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("choices", [{}])[0].get("text", "")
            if verbose:
                print(f"   ✓ Response: {text[:200]}")
        else:
            error_text = response.text[:500]
            if verbose:
                print(f"   ✗ HTTP {response.status_code}")
                print(f"   Error: {error_text}")
            return False, f"Long prompt failed: HTTP {response.status_code}: {error_text}", model_id
            
    except Exception as e:
        return False, f"Long prompt error: {e}", model_id
    
    # Step 5: Test with very long prompt (stress test)
    if verbose:
        print("\n5. Testing with very long prompt (stress test)...")
    
    very_long_prompt = long_prompt + "\n\n" + "Additional context. " * 500  # ~4K tokens
    
    try:
        payload = {
            "model": model_id,
            "prompt": very_long_prompt,
            "max_tokens": 50,
            "temperature": 0.7,
        }
        
        if verbose:
            print(f"   Prompt length: {len(very_long_prompt)} chars (~{len(very_long_prompt)//4} tokens)")
        
        response = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("choices", [{}])[0].get("text", "")
            if verbose:
                print(f"   ✓ Response: {text[:100]}")
        else:
            error_text = response.text[:500]
            if verbose:
                print(f"   ⚠ HTTP {response.status_code} (may be expected for very long prompts)")
                print(f"   Error: {error_text}")
            # This is not necessarily a failure - very long prompts may exceed limits
            
    except Exception as e:
        if verbose:
            print(f"   ⚠ Error (may be expected): {e}")
    
    return True, "All tests passed", model_id


def main():
    parser = argparse.ArgumentParser(description="Diagnose vLLM server issues")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Single port to test (default: test all)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    # Default ports for the training setup
    ports = {
        8001: "Orch-Question",
        8002: "Orch-DataEval", 
        8003: "Orch-Synthesis",
        8004: "Orch-AnswerEval",
        8005: "CypherGen",
    }
    
    if args.port:
        ports = {args.port: f"Server-{args.port}"}
    
    print("=" * 60)
    print("vLLM Server Diagnostics")
    print("=" * 60)
    
    results = {}
    for port, name in ports.items():
        success, message, model_id = test_server(args.host, port, not args.quiet)
        results[port] = {
            "name": name,
            "success": success,
            "message": message,
            "model_id": model_id,
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for port, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"  {status} Port {port} ({result['name']}): {result['message']}")
        if result["model_id"]:
            print(f"      Model ID: {result['model_id']}")
        if not result["success"]:
            all_ok = False
    
    if all_ok:
        print("\n✓ All servers are working correctly!")
    else:
        print("\n✗ Some servers have issues. Check the logs above.")
        print("\nTroubleshooting tips:")
        print("  1. Check vLLM server logs in the log directory")
        print("  2. Verify GPU memory is not exhausted (nvidia-smi)")
        print("  3. Ensure model path is correct and accessible")
        print("  4. Try restarting the problematic server")
    
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

