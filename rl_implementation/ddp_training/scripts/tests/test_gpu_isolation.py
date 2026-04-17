#!/usr/bin/env python3
"""
Test GPU isolation for vLLM workers.

This script tests that we can spawn multiple processes, each with their own
CUDA_VISIBLE_DEVICES, without conflicts.
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_worker(gpu_id: int, name: str, ready_event: mp.Event, error_queue: mp.Queue):
    """Test worker that initializes torch on a specific GPU."""
    # Set GPU BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    print(f"[{name}] Worker started (PID={os.getpid()})", flush=True)
    print(f"[{name}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        
        print(f"[{name}] torch.cuda.is_available()={cuda_available}", flush=True)
        print(f"[{name}] torch.cuda.device_count()={device_count}", flush=True)
        
        if cuda_available and device_count > 0:
            device_name = torch.cuda.get_device_name(0)
            print(f"[{name}] torch.cuda.get_device_name(0)={device_name}", flush=True)
            
            # Try to allocate some memory
            x = torch.zeros(1000, 1000, device='cuda:0')
            print(f"[{name}] ✓ Successfully allocated tensor on GPU", flush=True)
            del x
            torch.cuda.empty_cache()
            
            error_queue.put((name, None))
        else:
            error_queue.put((name, "CUDA not available"))
            
    except Exception as e:
        print(f"[{name}] ✗ Error: {e}", flush=True)
        error_queue.put((name, str(e)))
    finally:
        ready_event.set()


def main():
    print("=" * 60)
    print("GPU Isolation Test")
    print("=" * 60)
    print(f"Main process PID: {os.getpid()}")
    print(f"Main process CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print()
    
    # Disable CUDA in main process
    original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"Set CUDA_VISIBLE_DEVICES='' in main process")
    print()
    
    # Test GPUs
    test_gpus = [0, 1, 2, 3, 4, 5]
    
    ctx = mp.get_context('spawn')
    error_queue = ctx.Queue()
    
    workers = []
    events = []
    
    print("Starting workers...")
    for gpu_id in test_gpus:
        ready_event = ctx.Event()
        events.append(ready_event)
        
        name = f"Worker-GPU{gpu_id}"
        p = ctx.Process(
            target=test_worker,
            args=(gpu_id, name, ready_event, error_queue),
            daemon=True,
        )
        workers.append(p)
        p.start()
        print(f"  Started {name} (PID={p.pid})")
    
    print()
    print("Waiting for workers to complete...")
    
    # Wait for all workers
    for i, (event, worker) in enumerate(zip(events, workers)):
        if not event.wait(timeout=60):
            print(f"  ✗ Worker-GPU{test_gpus[i]} timed out")
        worker.join(timeout=5)
    
    print()
    print("Results:")
    
    # Check errors
    errors = []
    while True:
        try:
            name, error = error_queue.get_nowait()
            if error is not None:
                errors.append(f"{name}: {error}")
                print(f"  ✗ {name}: {error}")
            else:
                print(f"  ✓ {name}: Success")
        except:
            break
    
    print()
    if errors:
        print("=" * 60)
        print("FAILED - Some workers had errors")
        print("=" * 60)
        return 1
    else:
        print("=" * 60)
        print("SUCCESS - All workers initialized correctly")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())

