#!/usr/bin/env python3
"""
Monitor script for small-scale training test.

Monitors GPU usage, memory, and training progress during the test.

Usage:
    python -m rl_implementation.training.monitor_test
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def get_gpu_info():
    """Get GPU usage information."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) == 5:
                    gpus.append({
                        'index': parts[0],
                        'name': parts[1],
                        'utilization': f"{parts[2]}%",
                        'memory_used': f"{parts[3]} MB",
                        'memory_total': f"{parts[4]} MB"
                    })
            return gpus
        return None
    except Exception as e:
        return None


def get_ray_status():
    """Get Ray cluster status."""
    try:
        result = subprocess.run(
            ['ray', 'status'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


def tail_log_file(log_file, num_lines=10):
    """Get last N lines from log file."""
    try:
        if not os.path.exists(log_file):
            return None
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-num_lines:])
    except Exception:
        return None


def monitor_training(log_file='test_small_scale.log', interval=5, max_duration=600):
    """
    Monitor training progress.
    
    Args:
        log_file: Path to log file to monitor
        interval: Update interval in seconds
        max_duration: Maximum monitoring duration in seconds
    """
    print("=" * 80)
    print("TRAINING MONITOR")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print(f"Update interval: {interval}s")
    print(f"Max duration: {max_duration}s")
    print("=" * 80)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_duration:
                print(f"\n⏱️  Max duration ({max_duration}s) reached. Stopping monitor.")
                break
            
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Clear screen (optional - comment out if it causes issues)
            # os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"\n{'=' * 80}")
            print(f"[{timestamp}] Iteration {iteration} | Elapsed: {int(elapsed)}s")
            print("=" * 80)
            
            # GPU info
            gpus = get_gpu_info()
            if gpus:
                print("\n📊 GPU Status:")
                for gpu in gpus:
                    print(f"  GPU {gpu['index']} ({gpu['name']}): "
                          f"Util={gpu['utilization']}, "
                          f"Mem={gpu['memory_used']}/{gpu['memory_total']}")
            else:
                print("\n⚠️  Could not get GPU info")
            
            # Ray status
            ray_status = get_ray_status()
            if ray_status and 'Resources' in ray_status:
                print("\n🔷 Ray Cluster:")
                # Extract key info
                for line in ray_status.split('\n'):
                    if 'Resources' in line or 'GPU' in line or 'CPU' in line:
                        print(f"  {line.strip()}")
            
            # Log tail
            log_tail = tail_log_file(log_file, num_lines=5)
            if log_tail:
                print("\n📝 Recent Log (last 5 lines):")
                for line in log_tail.split('\n'):
                    if line.strip():
                        print(f"  {line}")
            else:
                print(f"\n⚠️  Log file not found: {log_file}")
            
            # Check for completion or errors
            if log_tail:
                if "TEST PASSED" in log_tail or "Training completed successfully" in log_tail:
                    print("\n✅ Training completed successfully!")
                    break
                elif "TEST FAILED" in log_tail or "ERROR" in log_tail:
                    print("\n❌ Error detected in training!")
                    print("\nLast 20 lines of log:")
                    full_tail = tail_log_file(log_file, num_lines=20)
                    if full_tail:
                        print(full_tail)
                    break
            
            print(f"\n{'=' * 80}")
            print(f"Next update in {interval}s... (Ctrl+C to stop)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
    
    print("\n" + "=" * 80)
    print("MONITORING COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor small-scale training test"
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='test_small_scale.log',
        help='Path to log file to monitor'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Update interval in seconds'
    )
    parser.add_argument(
        '--max-duration',
        type=int,
        default=600,
        help='Maximum monitoring duration in seconds'
    )
    
    args = parser.parse_args()
    
    monitor_training(
        log_file=args.log_file,
        interval=args.interval,
        max_duration=args.max_duration
    )


if __name__ == "__main__":
    main()

