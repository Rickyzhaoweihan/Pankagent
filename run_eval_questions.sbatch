#!/bin/bash
#SBATCH --job-name=eval_questions
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation/outputs/logs/eval_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation/outputs/logs/eval_%j.err

# =============================================================================
# Evaluation Questions Job
#
# This script:
#   1. Starts a vLLM server hosting the cypher-writer model on GPU
#   2. Waits for the server to be ready
#   3. Runs 7 evaluation questions through the multi-agent pipeline (main.py)
#   4. Logs each question + answer to outputs/eval_results.jsonl
#
# Usage:
#   sbatch run_eval_questions.sbatch
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "EVALUATION QUESTIONS JOB"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "============================================================"

# ----- Configuration -----
WORKDIR=/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation
MODEL_PATH=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/cypher_trained_hints_test_0118_2223/best_model_merged
VLLM_PORT=8002
OUTPUT_FILE="$WORKDIR/outputs/eval_results_rigor.jsonl"

# =============================================
# >>> SET YOUR ANTHROPIC API KEY HERE <<<
# =============================================
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY_HERE"
# =============================================

# Set to "true" to use rigor mode (strict evidence-only, shorter answers)
RIGOR_MODE="true"

cd "$WORKDIR"
mkdir -p outputs outputs/logs

export ANTHROPIC_API_KEY
export VLLM_PORT
export OPENAI_API_KEY="EMPTY"

# =============================================
# Phase 1: Start vLLM server
# =============================================
echo ""
echo "Phase 1: Starting vLLM cypher-writer server on port $VLLM_PORT..."
echo "============================================================"

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name cypher-writer \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-num-seqs 32 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=180
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready! (waited ${WAITED}s)"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  ... waiting (${WAITED}s / ${MAX_WAIT}s)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server did not start within ${MAX_WAIT}s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# =============================================
# Phase 2: Run evaluation questions
# =============================================
echo ""
echo "Phase 2: Running evaluation questions..."
echo "============================================================"

conda activate agent

# Clear previous results
> "$OUTPUT_FILE"

QUESTIONS=(
    "What is the expression of INS across cell types?"
    "What are the top 10 strongest differentially expressed genes in T1D?"
    "What are the beta cell regulatory elements for the gene PDX1?"
    "Which genes have formal colocalization with T1D GWAS signals?"
    "List the top some open chromatin regions that are unique to beta cells."
    "List the regulatory elements that regulate genes that are differentially expressed in T1D."
    "List the regulatory elements that regulate genes that are differentially expressed in T1D and provide papers that support the genes and regulatory elements and their involvement in T1D."
)

TOTAL=${#QUESTIONS[@]}
echo "Total questions: $TOTAL"
echo "Output file: $OUTPUT_FILE"
echo ""

for i in "${!QUESTIONS[@]}"; do
    Q="${QUESTIONS[$i]}"
    NUM=$((i + 1))
    echo "============================================================"
    echo "[$NUM/$TOTAL] $(date '+%H:%M:%S') - Question: $Q"
    echo "============================================================"

    START_TIME=$(date +%s)

    # Run main.py and capture the final_response event
    RIGOR_FLAG=""
    if [ "$RIGOR_MODE" = "true" ]; then
        RIGOR_FLAG="--rigor"
    fi
    RESPONSE=$(python3 main.py $RIGOR_FLAG "$Q" 2>&1 | tee /dev/stderr | grep '"event": "final_response"' | tail -1) || true

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "  Elapsed: ${ELAPSED}s"

    # Log as JSONL: one JSON object per line
    python3 -c "
import json, sys

q = sys.argv[1]
num = int(sys.argv[2])
elapsed = int(sys.argv[3])
raw_response = sys.argv[4] if len(sys.argv) > 4 else ''

entry = {
    'question_number': num,
    'question': q,
    'elapsed_seconds': elapsed,
}

# Try to parse the final_response event
if raw_response.strip():
    try:
        event = json.loads(raw_response)
        resp_data = event.get('data', {}).get('response', '')
        try:
            entry['answer'] = json.loads(resp_data)
        except (json.JSONDecodeError, TypeError):
            entry['answer'] = resp_data
    except json.JSONDecodeError:
        entry['answer'] = raw_response
else:
    entry['answer'] = None
    entry['error'] = 'No final_response event captured'

print(json.dumps(entry, ensure_ascii=False))
" "$Q" "$NUM" "$ELAPSED" "$RESPONSE" >> "$OUTPUT_FILE"

    echo "  Logged to $OUTPUT_FILE"
    echo ""
done

# =============================================
# Summary
# =============================================
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo "Results: $OUTPUT_FILE"
echo "Total questions: $TOTAL"
echo ""
echo "Per-question summary:"
python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    for line in f:
        entry = json.loads(line)
        num = entry['question_number']
        q = entry['question'][:80]
        elapsed = entry['elapsed_seconds']
        has_answer = entry.get('answer') is not None and entry.get('error') is None
        status = 'OK' if has_answer else 'FAIL'
        print(f'  Q{num}: [{status}] {elapsed:3d}s  {q}')
"
echo ""
echo "Finished: $(date)"

# =============================================
# Cleanup
# =============================================
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done."
