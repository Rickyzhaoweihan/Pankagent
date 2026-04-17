#!/usr/bin/env python3
"""
Test script for PlannerAgent API Server — tests both /query and /query/stream.

Usage:
    # Test against default port 8080
    python3 test_server_stream.py

    # Test against custom port
    python3 test_server_stream.py 9000

    # Test a specific question
    python3 test_server_stream.py 8080 "Is CFTR an effector gene for type 1 diabetes?"
"""

import json
import sys
import time
import requests

PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8080
QUESTION = sys.argv[2] if len(sys.argv) > 2 else "Is CFTR an effector gene for type 1 diabetes?"
BASE = f"http://localhost:{PORT}"


def test_health():
    print("=" * 60)
    print("1. GET /health")
    print("=" * 60)
    r = requests.get(f"{BASE}/health", timeout=5)
    print(f"   Status: {r.status_code}")
    print(f"   Body:   {r.json()}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print("   ✓ OK\n")


def test_query_sync():
    print("=" * 60)
    print(f"2. POST /query  (synchronous)")
    print(f"   Question: {QUESTION}")
    print("=" * 60)
    t0 = time.time()
    r = requests.post(
        f"{BASE}/query",
        json={"question": QUESTION},
        timeout=300,
    )
    elapsed = time.time() - t0
    print(f"   Status:  {r.status_code}")
    print(f"   Elapsed: {elapsed:.1f}s")
    if r.status_code == 200:
        data = r.json()
        print(f"   Server processing_time_ms: {data['processing_time_ms']:.0f}")
        answer = data["answer"]
        # Show first 500 chars
        print(f"   Answer (first 500 chars):\n")
        print("   " + answer[:500].replace("\n", "\n   "))
        if len(answer) > 500:
            print(f"   ... ({len(answer)} chars total)")
    else:
        print(f"   Error: {r.text[:500]}")
    print("   ✓ Done\n")


def test_query_stream():
    print("=" * 60)
    print(f"3. POST /query/stream  (streaming NDJSON)")
    print(f"   Question: {QUESTION}")
    print("=" * 60)
    t0 = time.time()
    r = requests.post(
        f"{BASE}/query/stream",
        json={"question": QUESTION},
        timeout=300,
        stream=True,  # important: stream the response
    )
    print(f"   Status: {r.status_code}")
    print(f"   Content-Type: {r.headers.get('content-type', '?')}")
    print()

    events = []
    final_answer = None

    # Read NDJSON lines as they arrive
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        elapsed = time.time() - t0
        try:
            event = json.loads(line)
            event_type = event.get("event", "?")
            data = event.get("data", {})
            events.append(event)

            # Print a compact summary of each event
            if event_type == "final_response":
                final_answer = data.get("response", "")
                print(f"   [{elapsed:6.1f}s] {event_type}  (answer: {len(final_answer)} chars)")
            elif event_type == "error":
                print(f"   [{elapsed:6.1f}s] ❌ {event_type}: {data.get('message', '')[:200]}")
            else:
                # Show a one-line summary
                summary = json.dumps(data, ensure_ascii=False)
                if len(summary) > 120:
                    summary = summary[:120] + "..."
                print(f"   [{elapsed:6.1f}s] {event_type:30s} {summary}")
        except json.JSONDecodeError:
            print(f"   [{elapsed:6.1f}s] (non-JSON line): {line[:100]}")

    total = time.time() - t0
    print()
    print(f"   Total events: {len(events)}")
    print(f"   Total time:   {total:.1f}s")

    if final_answer:
        print(f"\n   Final answer (first 500 chars):\n")
        print("   " + final_answer[:500].replace("\n", "\n   "))
        if len(final_answer) > 500:
            print(f"   ... ({len(final_answer)} chars total)")

    # Save all events to file for inspection
    out_file = "stream_results.json"
    with open(out_file, "w") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"\n   Saved {len(events)} events to {out_file}")
    print("   ✓ Done\n")


if __name__ == "__main__":
    print(f"\nTesting PlannerAgent API at {BASE}")
    print(f"Question: {QUESTION}\n")

    try:
        test_health()
        test_query_stream()
        # Uncomment to also test the synchronous endpoint:
        # test_query_sync()
    except requests.exceptions.ConnectionError:
        print(f"\n   ✗ Cannot connect to {BASE}")
        print(f"   Start the server first:  python3 server.py {PORT}")
    except KeyboardInterrupt:
        print("\n   Interrupted.")

