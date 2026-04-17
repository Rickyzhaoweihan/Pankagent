#!/usr/bin/env python3
"""
Simple test to verify GPU allocation is working correctly.

This script verifies that CUDA_VISIBLE_DEVICES is being respected
and that the model loads to the correct GPU.

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 rl_implementation/ddp_training/scripts/test_gpu_allocation.py
"""

import os
import sys

# CRITICAL: Print CUDA_VISIBLE_DEVICES before any imports
print("=" * 60)
print("Environment Check (before torch import)")
print("=" * 60)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print()

# Set default if not set
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    print("WARNING: CUDA_VISIBLE_DEVICES was not set, defaulting to '7'")
    print()

# Now import torch
import torch

print("=" * 60)
print("PyTorch CUDA Check")
print("=" * 60)
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    sys.exit(1)

# Check each visible GPU
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {props.name}")
    print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
    
    # Try to allocate and check memory
    torch.cuda.set_device(i)
    torch.cuda.empty_cache()
    
    free_before = torch.cuda.mem_get_info(i)[0] / 1024**3
    print(f"  Free before allocation: {free_before:.1f} GB")
    
    # Allocate a 1GB tensor to verify we can use this GPU
    try:
        x = torch.randn(256, 1024, 1024, device=f'cuda:{i}')  # ~1GB
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  Successfully allocated 1GB tensor")
        print(f"  Memory allocated: {allocated:.2f} GB")
        del x
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  FAILED: Out of memory - {e}")
    except Exception as e:
        print(f"  FAILED: {e}")

print()
print("=" * 60)
print("Model Loading Test (Small)")
print("=" * 60)

# Try to load a small model to verify the device mapping works
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Get model path from config
config_path = Path(__file__).parent.parent / "config" / "stage1_config.yaml"
if config_path.exists():
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_path = config.get('cypher_model_path', '')
    print(f"Model path from config: {model_path}")
else:
    print("Config not found, skipping model loading test")
    sys.exit(0)

if not model_path or not Path(model_path).exists():
    print(f"Model path does not exist: {model_path}")
    sys.exit(1)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("Tokenizer loaded successfully")

print()
print("Loading model (this may take a minute)...")
print("Note: Model should load to cuda:0 (which is physical GPU 7 if CUDA_VISIBLE_DEVICES=7)")

# Check memory before loading
free_before = torch.cuda.mem_get_info(0)[0] / 1024**3
print(f"Free memory before model load: {free_before:.1f} GB")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Load to cuda:0
        trust_remote_code=True,
    )
    
    # Check memory after loading
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"\nModel loaded successfully!")
    print(f"Memory allocated: {allocated:.2f} GB")
    print(f"Memory reserved: {reserved:.2f} GB")
    
    # Try a forward pass
    print("\nTesting forward pass...")
    inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model(**inputs)
    print("Forward pass successful!")
    print(f"Output logits shape: {outputs.logits.shape}")
    
    # Cleanup
    del model
    del outputs
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

