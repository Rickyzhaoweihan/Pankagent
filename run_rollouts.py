#!/usr/bin/env python3
"""
Batch Rollouts Evaluator

Reads questions from rollouts_iter_009_to_013_questions.txt,
sends each to the multi-agent system, and stores only the summary
in a JSON output file.

Supports resuming from the last completed question in case of crashes.

Usage:
    python run_rollouts.py
    python run_rollouts.py --input rollouts_iter_009_to_013_questions.txt --output rollouts_results.json
    python run_rollouts.py --start-from 50   # skip first 50 questions
"""

import json
import re
import os
import sys
import time
import signal
import argparse
import traceback
import multiprocessing
from datetime import datetime
from contextlib import contextmanager


@contextmanager
def suppress_stdout(log_file_path: str = None):
    """
    Context manager to suppress all stdout/stderr from agent internals.
    Optionally redirects to a log file for debugging.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    if log_file_path:
        devnull = open(log_file_path, 'a')
    else:
        devnull = open(os.devnull, 'w')
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


class QuestionTimeout(Exception):
    """Raised when a single question exceeds its time limit."""
    pass


def _run_question_in_process(question: str, result_queue: multiprocessing.Queue,
                             agent_log: str = None):
    """
    Worker function that runs a single question in a separate process.
    This ensures we can hard-kill it if it hangs.
    """
    try:
        # Re-import inside the child process
        from main import chat_single_round
        
        # Suppress agent output in the child process
        if agent_log:
            devnull = open(agent_log, 'a')
        else:
            devnull = open(os.devnull, 'w')
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        
        try:
            response = chat_single_round(question)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()
        
        result_queue.put(("success", response))
    except Exception as e:
        result_queue.put(("error", str(e)))


def run_question_with_timeout(question: str, timeout: int = 300,
                              agent_log: str = None) -> str:
    """
    Run a single question with a hard timeout using a subprocess.
    If the question takes longer than `timeout` seconds, the subprocess
    is killed and a QuestionTimeout is raised.
    """
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_question_in_process,
        args=(question, result_queue, agent_log)
    )
    proc.start()
    proc.join(timeout=timeout)
    
    if proc.is_alive():
        # Hard kill the hanging process
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        raise QuestionTimeout(
            f"Question timed out after {timeout}s: {question[:80]}..."
        )
    
    if result_queue.empty():
        raise RuntimeError("Agent process exited without returning a result")
    
    status, payload = result_queue.get_nowait()
    if status == "error":
        raise RuntimeError(payload)
    return payload


def parse_questions(filepath: str) -> list:
    """
    Parse the rollouts questions file into a structured list.
    
    Returns:
        List of dicts: [{"iter": "009", "index": 1, "question": "..."}, ...]
    """
    questions = []
    current_iter = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Match section headers like "=== rollouts_iter_009 (168 questions) ==="
            header_match = re.match(r'^=== rollouts_iter_(\d+)\s+\(\d+ questions\)\s*===', line)
            if header_match:
                current_iter = header_match.group(1)
                continue
            
            # Match question lines like "1. Which genes are..."
            question_match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if question_match and current_iter:
                idx = int(question_match.group(1))
                question_text = question_match.group(2).strip()
                if question_text:
                    questions.append({
                        "iter": current_iter,
                        "index": idx,
                        "question": question_text
                    })
    
    return questions


def load_existing_results(output_path: str) -> dict:
    """Load existing results for resumption support."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, KeyError):
            return {"metadata": {}, "results": []}
    return {"metadata": {}, "results": []}


def extract_summary(format_result: str) -> str:
    """Extract only the summary text from the FormatAgent/ReasoningAgent JSON response."""
    try:
        result_json = json.loads(format_result)
        
        # Handle template_matching case
        if "template_matching" in result_json and "text" not in result_json:
            return result_json.get("template_matching", "")
        
        # Standard format: {"to": "user", "text": {"summary": "...", ...}}
        text = result_json.get("text", {})
        if isinstance(text, dict):
            return text.get("summary", "")
        elif isinstance(text, str):
            return text
        
        return str(result_json)
    except json.JSONDecodeError:
        return format_result


def extract_reasoning_trace(format_result: str) -> str:
    """Extract the reasoning trace from the ReasoningAgent JSON response (if present)."""
    try:
        result_json = json.loads(format_result)
        text = result_json.get("text", {})
        if isinstance(text, dict):
            return text.get("reasoning_trace", "")
        return ""
    except (json.JSONDecodeError, KeyError):
        return ""


def extract_cypher(format_result: str) -> list:
    """Extract cypher queries from the FormatAgent JSON response."""
    try:
        result_json = json.loads(format_result)
        text = result_json.get("text", {})
        if isinstance(text, dict):
            return text.get("cypher", [])
        return []
    except (json.JSONDecodeError, KeyError):
        return []


def extract_full_response(format_result: str) -> dict:
    """Extract the full parsed response."""
    try:
        return json.loads(format_result)
    except json.JSONDecodeError:
        return {"raw": format_result}


def wait_for_vllm(host: str = "localhost", port: int = 8001,
                  timeout: int = 600, interval: int = 10) -> bool:
    """Wait for the vLLM server to become ready."""
    import urllib.request
    import urllib.error
    
    url = f"http://{host}:{port}/health"
    start = time.time()
    
    print(f"Waiting for vLLM server at {host}:{port} ...")
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                print(f"✅ vLLM server is ready (took {time.time()-start:.0f}s)")
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(interval)
    
    print(f"❌ vLLM server did not start within {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run rollout questions through the multi-agent system")
    parser.add_argument('--input', type=str,
                        default='rollouts_iter_009_to_013_questions.txt',
                        help='Path to the questions file')
    parser.add_argument('--output', type=str,
                        default='rollouts_results.json',
                        help='Path to the output JSON file')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Skip the first N questions (for manual resumption)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process N questions (for testing)')
    parser.add_argument('--skip-vllm-check', action='store_true',
                        help='Skip waiting for vLLM server')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from last completed question (default: True)')
    parser.add_argument('--agent-log', type=str, default=None,
                        help='Path to dump agent debug output (default: suppress entirely)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Per-question timeout in seconds (default: 300 = 5 min)')
    
    args = parser.parse_args()
    
    # Where to send agent chatter (None = /dev/null)
    agent_log = args.agent_log
    question_timeout = args.timeout
    
    # Parse all questions
    print("=" * 60)
    print("ROLLOUTS BATCH EVALUATOR")
    print("=" * 60)
    
    questions = parse_questions(args.input)
    print(f"Parsed {len(questions)} questions from {args.input}")
    
    # Count by iteration
    iter_counts = {}
    for q in questions:
        iter_counts[q['iter']] = iter_counts.get(q['iter'], 0) + 1
    for it, count in sorted(iter_counts.items()):
        print(f"  rollouts_iter_{it}: {count} questions")
    
    # Wait for vLLM server
    if not args.skip_vllm_check:
        if not wait_for_vllm():
            print("ERROR: vLLM server is not available. Exiting.")
            sys.exit(1)
    
    # Load existing results for resumption
    existing_data = load_existing_results(args.output)
    completed_keys = set()
    if args.resume and existing_data.get("results"):
        for r in existing_data["results"]:
            key = f"iter_{r['iter']}_q{r['index']}"
            completed_keys.add(key)
        print(f"Found {len(completed_keys)} already-completed questions, will skip them")
    
    # Prepare results storage
    results = existing_data.get("results", [])
    
    # Apply start-from offset
    questions_to_process = questions[args.start_from:]
    if args.limit:
        questions_to_process = questions_to_process[:args.limit]
    
    total = len(questions_to_process)
    success_count = len(completed_keys)
    error_count = 0
    start_time = time.time()
    
    print(f"\nProcessing {total} questions (timeout: {question_timeout}s per question)...")
    print(f"Output will be saved to: {args.output}")
    print("=" * 60)
    sys.stdout.flush()
    
    for i, q in enumerate(questions_to_process):
        # Check if already completed
        key = f"iter_{q['iter']}_q{q['index']}"
        if key in completed_keys:
            continue
        
        global_idx = args.start_from + i + 1
        question = q['question']
        
        q_start = time.time()
        
        try:
            # Run in a separate process with hard timeout
            response = run_question_with_timeout(
                question, timeout=question_timeout, agent_log=agent_log
            )
            q_elapsed = time.time() - q_start
            
            # Extract summary and optional reasoning trace
            summary = extract_summary(response)
            cypher = extract_cypher(response)
            reasoning_trace = extract_reasoning_trace(response)
            
            result_entry = {
                "iter": q['iter'],
                "index": q['index'],
                "question": question,
                "summary": summary,
                "cypher": cypher,
                "elapsed_seconds": round(q_elapsed, 2),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            # Include reasoning trace if present (complex questions)
            if reasoning_trace:
                result_entry["reasoning_trace"] = reasoning_trace
            
            results.append(result_entry)
            success_count += 1
            
            print(f"✅ [{global_idx}/{len(questions)}] {q_elapsed:.1f}s | {question[:60]}")
            sys.stdout.flush()
            
        except QuestionTimeout as e:
            q_elapsed = time.time() - q_start
            
            result_entry = {
                "iter": q['iter'],
                "index": q['index'],
                "question": question,
                "summary": "",
                "cypher": [],
                "elapsed_seconds": round(q_elapsed, 2),
                "status": "timeout",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result_entry)
            error_count += 1
            
            print(f"⏰ [{global_idx}/{len(questions)}] TIMEOUT {q_elapsed:.0f}s | {question[:60]}")
            sys.stdout.flush()
            
        except Exception as e:
            q_elapsed = time.time() - q_start
            
            result_entry = {
                "iter": q['iter'],
                "index": q['index'],
                "question": question,
                "summary": "",
                "cypher": [],
                "elapsed_seconds": round(q_elapsed, 2),
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result_entry)
            error_count += 1
            
            print(f"❌ [{global_idx}/{len(questions)}] {q_elapsed:.1f}s | {str(e)[:100]}")
            sys.stdout.flush()
        
        # Save after every question (for crash recovery)
        output_data = {
            "metadata": {
                "input_file": args.input,
                "total_questions": len(questions),
                "processed": len(results),
                "successes": success_count,
                "errors": error_count,
                "last_updated": datetime.now().isoformat(),
                "total_elapsed_seconds": round(time.time() - start_time, 2)
            },
            "results": results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Final summary
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"Processed:       {len(results)}")
    print(f"Successes:       {success_count}")
    print(f"Errors:          {error_count}")
    print(f"Total time:      {total_elapsed/3600:.1f} hours ({total_elapsed:.0f}s)")
    if success_count > 0:
        print(f"Avg per question: {total_elapsed/max(success_count,1):.1f}s")
    print(f"Output saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    # Use 'spawn' to avoid issues with forked GPU/API connections
    multiprocessing.set_start_method('spawn', force=True)
    main()

