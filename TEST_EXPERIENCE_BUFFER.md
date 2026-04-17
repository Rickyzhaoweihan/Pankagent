# Testing the RAG Experience Buffer System

This guide walks through testing each component of the experience buffer system.

## Quick Test Checklist

- [ ] Experience buffer module loads correctly
- [ ] Server logs planning data
- [ ] Batch evaluator processes logs
- [ ] Experience buffer is loaded on startup
- [ ] Examples are injected into prompts

---

## Test 1: Experience Buffer Module

Test that the core module works:

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation

# Test the module directly
python3 -c "
from PankBaseAgent.experience_buffer import ExperienceBuffer

# Create buffer
buffer = ExperienceBuffer()

# Test logging
buffer.log_planning(
    query='Test query about CFTR',
    planning={'num_queries': 3, 'queries': [{'name': 'pankbase_api_query', 'input': 'Find gene CFTR'}]},
    results={'success': True, 'data_count': 1},
    execution_time_ms=1234.5
)

# Test stats
stats = buffer.get_stats()
print('Stats:', stats)

# Test pattern detection
pattern = buffer._detect_pattern('Tell me about gene CFTR')
print('Pattern detected:', pattern)

print('✓ Experience buffer module works!')
"
```

**Expected output:**
```
Stats: {'total_logged': 1, 'total_curated': 0, 'log_file_exists': True, 'buffer_file_exists': False}
Pattern detected: entity_overview
✓ Experience buffer module works!
```

**Verify:**
- [ ] File `logs/query_log.jsonl` was created
- [ ] Contains one JSON entry

---

## Test 2: Server Logging

Test that the server logs planning data:

```bash
# Start the server
python3 server.py 8080 &
SERVER_PID=$!

# Wait for startup
sleep 5

# Send a test query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about gene CFTR"}'

# Wait for background logging
sleep 2

# Check the log file
echo "Checking query log..."
tail -1 logs/query_log.jsonl | python3 -m json.tool

# Stop server
kill $SERVER_PID
```

**Expected output:**
- JSON entry with fields: `timestamp`, `query`, `planning`, `results`, `execution_time_ms`
- `planning` should contain: `num_queries`, `queries`, `draft`

**Verify:**
- [ ] `logs/query_log.jsonl` contains the query
- [ ] Planning data includes number of queries
- [ ] Timestamp is recent

---

## Test 3: Planning Data Collection

Test that planning data is captured from PankBaseAgent:

```bash
# Run a query and check console output
python3 -c "
from main import chat_single_round

response = chat_single_round('What is the relationship between CFTR and Type 1 Diabetes?')
print('Response received')

# Check if planning data was collected
from utils import get_all_planning_data
planning_data = get_all_planning_data()

print(f'\nPlanning data collected: {len(planning_data)} entries')
for i, p in enumerate(planning_data, 1):
    print(f'  Entry {i}: {p.get(\"num_queries\", 0)} queries')
    print(f'    Draft: {p.get(\"draft\", \"N/A\")[:50]}...')
"
```

**Expected output:**
```
************************************************************
PANKBASE AGENT PLANNING (Round 1):
************************************************************
Draft/Plan: [some planning text]

Number of sub-queries planned: [number]
...
************************************************************

Response received

Planning data collected: 1 entries
  Entry 1: [number] queries
    Draft: [planning draft text]...
```

**Verify:**
- [ ] Console shows "PANKBASE AGENT PLANNING"
- [ ] Planning data was collected
- [ ] Draft and queries are captured

---

## Test 4: Batch Evaluator

Test the GPT-4 evaluation script:

**Prerequisites:**
- At least one query logged in `logs/query_log.jsonl`
- `OPENAI_API_KEY` environment variable set

```bash
# Run batch evaluator on small sample
python3 batch_evaluator.py --limit 5

# Check output
echo "Checking experience buffer..."
cat experience_buffer.jsonl | python3 -m json.tool | head -30
```

**Expected output:**
```
============================================================
Batch Experience Buffer Evaluator
============================================================

Loading queries from logs/query_log.jsonl...
Found [N] new queries to evaluate
Limiting to 5 queries

Evaluating queries in batches of 50...
Evaluating batch 1 (5 queries)...
  ✓ Evaluated 5 queries

✓ Evaluated 5 queries

Extracting top patterns...
Selected [N] top patterns

Top 5 patterns:
  1. Rating 8/10 - entity_overview: "Tell me about gene CFTR..."
  ...

Updating experience_buffer.jsonl...
✓ Updated experience buffer: [N] patterns (added [N] new)

✓ Updated last processed timestamp: 2025-01-18T...
============================================================
Evaluation complete!
============================================================
```

**Verify:**
- [ ] `experience_buffer.jsonl` was created
- [ ] Contains JSON entries with `rating`, `pattern`, `feedback`
- [ ] `logs/last_evaluated.txt` was created

---

## Test 5: Experience Buffer Loading on Startup

Test that the server loads examples:

```bash
# Make sure experience_buffer.jsonl exists (run Test 4 first)

# Start server and check logs
python3 server.py 8080 2>&1 | grep -A 5 "Loading experience buffer"
```

**Expected output:**
```
Loading experience buffer...
✓ Experience buffer loaded: [N] examples from [N] curated
```

**Verify:**
- [ ] Server logs show "Loading experience buffer"
- [ ] Shows number of examples loaded
- [ ] No errors during loading

---

## Test 6: Example Injection into Prompts

Test that examples are injected into PankBaseAgent prompts:

```bash
# Enable detailed logging
export LOG_PANKBASE_PROMPTS=1

# Run a query similar to one in experience buffer
python3 -c "
import sys
sys.path.append('PankBaseAgent')

from PankBaseAgent.ai_assistant import chat_one_round_pankbase

# Simulate server loading examples
import server
from PankBaseAgent.experience_buffer import get_experience_buffer
buffer = get_experience_buffer()
server.loaded_experiences = buffer.load_best_examples(max_examples=50)

print(f'Loaded {len(server.loaded_experiences)} examples')

# Run a query
messages, response, planning = chat_one_round_pankbase([], 'Tell me about gene INS')

print('Query completed')
"
```

**Check the PankBaseAgent logs:**
```bash
tail -100 PankBaseAgent/logs/claude_log.txt | grep -A 20 "LEARNED PATTERNS"
```

**Expected output:**
- Should see "LEARNED PATTERNS" section in the prompt
- Should show 1-3 similar examples
- Examples should have ratings and feedback

**Verify:**
- [ ] Prompt contains "LEARNED PATTERNS" section
- [ ] Shows relevant examples (similar queries)
- [ ] Examples include ratings and feedback

---

## Test 7: End-to-End Integration Test

Full integration test with the server:

```bash
# 1. Clear logs for fresh test
rm -f logs/query_log.jsonl logs/last_evaluated.txt experience_buffer.jsonl

# 2. Start server
python3 server.py 8080 &
SERVER_PID=$!
sleep 5

# 3. Send multiple queries
echo "Sending test queries..."
for query in \
  "Tell me about gene CFTR" \
  "What is the relationship between INS and Type 1 Diabetes?" \
  "Find genes that are effector genes for Type 1 Diabetes"
do
  curl -s -X POST http://localhost:8080/query \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$query\"}" > /dev/null
  echo "  ✓ Sent: $query"
  sleep 2
done

# 4. Check logs
echo -e "\n✓ Queries logged: $(wc -l < logs/query_log.jsonl)"

# 5. Run evaluator
echo -e "\nRunning batch evaluator..."
python3 batch_evaluator.py --limit 10

# 6. Check experience buffer
echo -e "\n✓ Experience buffer entries: $(wc -l < experience_buffer.jsonl)"

# 7. Restart server to load examples
kill $SERVER_PID
sleep 2
python3 server.py 8080 &
SERVER_PID=$!
sleep 5

# 8. Send another query (should use examples)
echo -e "\nSending query with experience buffer..."
curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about gene PNPLA3"}' | python3 -m json.tool

# Cleanup
kill $SERVER_PID
```

**Verify:**
- [ ] All 3 queries were logged
- [ ] Batch evaluator processed them
- [ ] Experience buffer was created
- [ ] Server loaded examples on restart
- [ ] Final query used examples (check logs)

---

## Test 8: Monitoring Dashboard

Test the monitoring tool:

```bash
# View summary
python3 view_experience.py

# View top examples
python3 view_experience.py --top 5

# View detailed stats
python3 view_experience.py --stats

# List patterns
python3 view_experience.py --list-patterns

# View specific pattern
python3 view_experience.py --pattern entity_overview
```

**Expected output:**
- Summary shows total queries logged and curated
- Top examples show ratings and queries
- Stats show pattern distribution
- Pattern-specific view shows relevant examples

**Verify:**
- [ ] All commands run without errors
- [ ] Shows meaningful statistics
- [ ] Data matches what's in files

---

## Troubleshooting

### Issue: No planning data collected

**Check:**
```bash
# Verify PankBaseAgent is being called
grep "PANKBASE AGENT PLANNING" logs/*.log

# Check if planning data is in global variable
python3 -c "from utils import get_all_planning_data; print(len(get_all_planning_data()))"
```

### Issue: Experience buffer not loading

**Check:**
```bash
# Verify file exists
ls -lh experience_buffer.jsonl

# Verify JSON is valid
python3 -c "
import json
with open('experience_buffer.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except:
            print(f'Invalid JSON on line {i}')
"
```

### Issue: Examples not injected

**Check:**
```bash
# Verify loaded_experiences is populated
python3 -c "
import server
from PankBaseAgent.experience_buffer import get_experience_buffer
buffer = get_experience_buffer()
server.loaded_experiences = buffer.load_best_examples()
print(f'Loaded: {len(server.loaded_experiences)} examples')
"

# Check if server module is accessible from PankBaseAgent
python3 -c "
import sys
sys.path.append('PankBaseAgent')
import sys
print('server' in sys.modules)
"
```

---

## Success Criteria

The system is working correctly if:

✅ **Logging**: Queries are logged to `logs/query_log.jsonl` with planning data  
✅ **Evaluation**: Batch evaluator processes logs and creates `experience_buffer.jsonl`  
✅ **Loading**: Server loads examples on startup (check logs)  
✅ **Injection**: Prompts contain "LEARNED PATTERNS" section (check PankBaseAgent logs)  
✅ **Improvement**: Over time, planning quality improves (higher num_queries, better patterns)

---

## Performance Monitoring

Track improvement over time:

```bash
# Average number of queries over time
python3 -c "
import json
with open('logs/query_log.jsonl') as f:
    queries = [json.loads(line) for line in f if line.strip()]

# Group by date
from collections import defaultdict
by_date = defaultdict(list)
for q in queries:
    date = q['timestamp'][:10]
    by_date[date].append(q['planning']['num_queries'])

for date in sorted(by_date.keys()):
    avg = sum(by_date[date]) / len(by_date[date])
    print(f'{date}: avg {avg:.1f} queries per request')
"
```

Expected: Average number of queries should increase as the system learns to be more comprehensive.

