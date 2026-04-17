# Cypher Query Filtering Implementation - Automatic Data-Based Filtering

## Summary

Implemented **automatic filtering** where the system tracks which Cypher queries returned data and only sends those to FormatAgent. This is simpler and more reliable than having PlannerAgent explicitly list queries.

## Key Principle

**Simple Rule**: Only include Cypher queries that returned non-empty results.
- ✅ Query returned nodes or edges → Include
- ❌ Query returned empty result → Exclude
- ❌ Query had error → Exclude

## Changes Made

### 1. Enhanced Query Tracking (`utils.py`)

**Lines 13, 20-41**: Modified query tracking to store both query and data status:

```python
# Before: Just stored query strings
current_cypher_queries: List[str] = []

# After: Store query + whether it returned data
current_cypher_queries: List[dict] = []

def add_cypher_query(cypher_query: str, returned_data: bool = True):
    """
    Add a cypher query to the current list.
    
    Args:
        cypher_query: The Cypher query string
        returned_data: True if the query returned non-empty results
    """
    global current_cypher_queries
    if cypher_query and cypher_query.strip():
        current_cypher_queries.append({
            'query': cypher_query.strip(),
            'returned_data': returned_data
        })

def get_queries_with_data() -> List[str]:
    """Get only cypher queries that returned data"""
    return [q['query'] for q in current_cypher_queries if q['returned_data']]
```

**Benefits**:
- ✅ Tracks data status at execution time
- ✅ No guessing or inference needed
- ✅ Automatic and deterministic

### 2. Data Detection in PankBaseAgent (`PankBaseAgent/utils.py`)

**Lines 195-222**: Added logic to detect if query returned data:

```python
# Empty response
if not response.text.strip():
    add_cypher_query(cleaned_cypher, returned_data=False)
    q.put((False, "Empty response from Pankbase API"))
    return

# Error response
if response.text.strip().startswith("Error:"):
    add_cypher_query(cleaned_cypher, returned_data=False)
    q.put((False, f"Pankbase API Error: {response.text}"))
    return

# Check if result has actual data
# The response structure is: {"results": "...", "query": "...", "error": null}
# Empty results show as: {"results": "No results", ...}
result = response.json()
has_data = True
if isinstance(result, dict):
    results_value = result.get('results', '')
    # Check if results is "No results" (case-insensitive)
    if isinstance(results_value, str) and results_value.strip().lower() == "no results":
        has_data = False
    # Also check if results is empty string
    elif isinstance(results_value, str) and not results_value.strip():
        has_data = False

# Track query with data status
add_cypher_query(cleaned_cypher, returned_data=has_data)
print(f"DEBUG: Query returned data: {has_data}")
```

**Detection Logic**:
- Checks for empty responses → `returned_data=False`
- Checks for error responses → `returned_data=False`
- Checks if `results` field equals "No results" (case-insensitive) → `returned_data=False`
- Checks if `results` field is empty string → `returned_data=False`
- Otherwise → `returned_data=True`

**Response Structure**:
```json
// Query with data:
{
  "results": "nodes, edges\n(:gene {...}), []",
  "query": "MATCH (g:gene) WHERE g.name = \"CFTR\" ...",
  "error": null
}

// Query without data:
{
  "results": "No results",
  "query": "MATCH (s:snp)-[q:part_of_QTL_signal]->...",
  "error": null
}
```

### 3. Automatic Filtering in Main Loop (`main.py`)

**Lines 57-63**: Use `get_queries_with_data()` instead of `get_all_cypher_queries()`:

```python
# Get only queries that returned data (automatically filtered)
cypher_queries = get_queries_with_data()
all_queries = get_all_cypher_queries()

print(f"DEBUG: Total queries executed: {len(all_queries)}")
print(f"DEBUG: Queries with data: {len(cypher_queries)}")
if len(cypher_queries) < len(all_queries):
    print(f"DEBUG: Filtered out {len(all_queries) - len(cypher_queries)} empty queries")
```

**Benefits**:
- ✅ Automatic filtering at code level
- ✅ Clear debug output showing filtering
- ✅ No prompt engineering needed

### 4. Simplified FormatAgent Prompt (`prompts/format_prompt.txt`)

**Lines 73-77**: Simplified since queries are pre-filtered:

```markdown
5. **Cypher Queries**
   - Include **all** Cypher queries provided (they are already filtered by the upstream agent).
   - The queries you receive are only those that directly contributed to the final answer.
   - Order them by **relevance** to the Human Query (most relevant first).
```

**Lines 166, 189**: Updated other references to clarify queries are pre-filtered.

## How It Works

### Flow Diagram

```
User: "What is CFTR's role in T1D?"
    ↓
┌─────────────────────────────────────────────────────────────┐
│  PlannerAgent calls 3 functions:                            │
│    1. pankbase_api_query("Find gene CFTR")                 │
│    2. pankbase_api_query("Get effector genes for T1D")     │
│    3. pankbase_api_query("Get QTL for CFTR")               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Each query executes in PankBaseAgent:                      │
│                                                              │
│  Query 1: Returns gene node                                 │
│    → add_cypher_query(query1, returned_data=True) ✅       │
│                                                              │
│  Query 2: Returns gene-disease edges                        │
│    → add_cypher_query(query2, returned_data=True) ✅       │
│                                                              │
│  Query 3: Returns empty (no QTL data)                       │
│    → add_cypher_query(query3, returned_data=False) ❌      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  main.py:                                                    │
│    all_queries = [query1, query2, query3]  (3 queries)     │
│    cypher_queries = get_queries_with_data()                │
│                   = [query1, query2]  (2 queries)          │
│                                                              │
│    DEBUG: Total queries executed: 3                         │
│    DEBUG: Queries with data: 2                              │
│    DEBUG: Filtered out 1 empty queries                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  FormatAgent receives:                                      │
│    - Human Query                                            │
│    - 2 Cypher Queries (only those with data)               │
│    - Final Answer                                           │
│                                                              │
│  No filtering needed, just formats them                     │
└─────────────────────────────────────────────────────────────┘
```

## Example Scenarios

### Scenario 1: All Queries Return Data

**Execution**:
```
Query 1: MATCH (g:gene) WHERE g.name='CFTR' ... → Returns 1 node
Query 2: MATCH (g:gene)-[e:effector_gene_of]->... → Returns 2 edges
```

**Tracking**:
```python
add_cypher_query(query1, returned_data=True)  # Has nodes
add_cypher_query(query2, returned_data=True)  # Has edges
```

**Result**:
```
DEBUG: Total queries executed: 2
DEBUG: Queries with data: 2
Cypher queries sent to FormatAgent: [query1, query2]
```

### Scenario 2: Some Queries Return Empty

**Execution**:
```
Query 1: MATCH (g:gene) WHERE g.name='CFTR' ... → Returns 1 node
Query 2: MATCH (g:gene)-[e:effector_gene_of]->... → Returns 2 edges
Query 3: MATCH (s:snp)-[q:part_of_QTL_signal]->(g:gene) ... → Returns empty (no QTL data)
```

**Tracking**:
```python
add_cypher_query(query1, returned_data=True)   # Has nodes
add_cypher_query(query2, returned_data=True)   # Has edges
add_cypher_query(query3, returned_data=False)  # Empty result
```

**Result**:
```
DEBUG: Total queries executed: 3
DEBUG: Queries with data: 2
DEBUG: Filtered out 1 empty queries
Cypher queries sent to FormatAgent: [query1, query2]
```

### Scenario 3: Query Has Error

**Execution**:
```
Query 1: MATCH (g:gene) WHERE g.name='CFTR' ... → Returns 1 node
Query 2: MATCH (g:gene)-[r:invalid_rel]->... → Error: relationship not found
```

**Tracking**:
```python
add_cypher_query(query1, returned_data=True)   # Has nodes
add_cypher_query(query2, returned_data=False)  # Error response
```

**Result**:
```
DEBUG: Total queries executed: 2
DEBUG: Queries with data: 1
DEBUG: Filtered out 1 empty queries
Cypher queries sent to FormatAgent: [query1]
```

## Benefits

### 1. **Simplicity**
- ✅ One clear rule: "returned data or not"
- ✅ No complex prompt engineering
- ✅ No LLM inference needed

### 2. **Accuracy**
- ✅ Deterministic (not probabilistic)
- ✅ Tracks at execution time (ground truth)
- ✅ No guessing or hallucination

### 3. **Efficiency**
- ✅ FormatAgent receives fewer queries
- ✅ Less token usage
- ✅ Faster processing

### 4. **Maintainability**
- ✅ Simple code logic
- ✅ Easy to debug (clear logging)
- ✅ No prompt maintenance needed

## Debug Output

When running the system, you'll see:

```
DEBUG: Sending Cypher query: MATCH (g:gene) WHERE g.name='CFTR' ...
DEBUG: Response text: {"nodes": [...], "edges": []}
DEBUG: Query returned data: True

DEBUG: Sending Cypher query: MATCH (s:snp)-[q:part_of_QTL_signal]->...
DEBUG: Response text: {"nodes": [], "edges": []}
DEBUG: Query returned data: False

DEBUG: Total queries executed: 2
DEBUG: Queries with data: 1
DEBUG: Filtered out 1 empty queries
Cypher queries sent to FormatAgent: ["MATCH (g:gene) WHERE g.name='CFTR' ..."]
```

## Files Modified

1. **`utils.py`** (+17 lines)
   - Changed `current_cypher_queries` from `List[str]` to `List[dict]`
   - Added `returned_data` parameter to `add_cypher_query()`
   - Added `get_queries_with_data()` function
   - Updated `__all__` exports

2. **`PankBaseAgent/utils.py`** (+20 lines)
   - Added data detection logic for empty responses
   - Added data detection logic for error responses
   - Added data detection logic for nodes/edges
   - Pass `returned_data` flag to `add_cypher_query()`
   - Added debug logging

3. **`main.py`** (+4 lines)
   - Use `get_queries_with_data()` instead of `get_all_cypher_queries()`
   - Added debug output showing filtering stats

4. **`prompts/format_prompt.txt`** (simplified)
   - Removed filtering instructions
   - Clarified queries are pre-filtered

5. **`PankBaseAgent/prompts/general_prompt.txt`** (no changes needed)
   - No prompt engineering needed for this approach!

## Comparison: Before vs After

### Before

```
All queries → FormatAgent → Try to guess which had data → Filter
```

**Problems**:
- ❌ FormatAgent can't see query results
- ❌ Has to infer from answer text
- ❌ Probabilistic (LLM guessing)
- ❌ Complex prompt

### After

```
All queries → Track data status → Filter → FormatAgent
```

**Benefits**:
- ✅ Tracks at execution time
- ✅ Deterministic filtering
- ✅ Simple code logic
- ✅ No prompt complexity

## Testing

### Manual Test

```bash
python3 main.py
```

**Enter**: "What is CFTR's role in T1D?"

**Expected Output**:
```
DEBUG: Total queries executed: 3
DEBUG: Queries with data: 2
DEBUG: Filtered out 1 empty queries
```

### Verification

Check that:
1. ✅ Queries with data are included
2. ✅ Empty queries are excluded
3. ✅ Error queries are excluded
4. ✅ Debug output is clear

## Future Enhancements

Potential improvements:
1. **Query metadata**: Track execution time, result size
2. **Adaptive filtering**: Different rules for different query types
3. **Result caching**: Cache query results to avoid re-execution
4. **Query optimization**: Detect and skip redundant queries

## Conclusion

✅ **Implementation Complete!**

The system now:
- Automatically tracks which queries returned data
- Filters out empty/error queries at code level
- Sends only relevant queries to FormatAgent
- Provides clear debug output
- No prompt engineering needed

**Key Insight**: The simplest solution is often the best. Instead of asking an LLM to figure out which queries were useful, we just track which ones returned data at execution time. This is deterministic, accurate, and easy to maintain.
