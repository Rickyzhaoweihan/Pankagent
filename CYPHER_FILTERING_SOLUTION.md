# Solution: Filter Cypher Queries for FormatAgent

## Problem

Currently, `get_all_cypher_queries()` returns ALL Cypher queries generated during the conversation, including:
- ❌ Failed queries from refinement iterations
- ❌ Exploratory queries that didn't contribute to the answer
- ❌ Duplicate queries

**You only want**: ✅ The FINAL successful queries that actually contributed to the answer

## Current Flow

```
User Question
    ↓
PankBaseAgent calls pankbase_api_query multiple times
    ↓
Each call:
  1. Text2CypherAgent generates Cypher (iteration 1)
     → add_cypher_query(cypher1)  ❌ Added
  2. If score < 95, refinement (iteration 2)
     → add_cypher_query(cypher2)  ❌ Added
  3. If still low, refinement (iteration 3)
     → add_cypher_query(cypher3)  ❌ Added
  4. Final query used
     → add_cypher_query(final_cypher)  ❌ Added
    ↓
FormatAgent receives ALL queries (cypher1, cypher2, cypher3, final_cypher)
```

**Problem**: FormatAgent gets 4 queries when only 1 (final_cypher) was actually used!

## Solution: Track Only Successful Queries

### Option 1: Add Success Flag (Recommended)

**Modify `utils.py`** to track which queries were successful:

```python
# Global variables
current_cypher_queries: List[dict] = []  # Changed from List[str]

def reset_cypher_queries():
    """Reset the cypher queries list for a new human query"""
    global current_cypher_queries
    current_cypher_queries = []

def add_cypher_query(cypher_query: str, is_final: bool = False):
    """
    Add a cypher query to the current list.
    
    Args:
        cypher_query: The Cypher query string
        is_final: True if this is the final successful query that will be executed
    """
    global current_cypher_queries
    if cypher_query and cypher_query.strip():
        current_cypher_queries.append({
            'query': cypher_query.strip(),
            'is_final': is_final
        })

def get_final_cypher_queries() -> List[str]:
    """Get only the final successful cypher queries"""
    return [q['query'] for q in current_cypher_queries if q['is_final']]

def get_all_cypher_queries() -> List[str]:
    """Get all cypher queries (for debugging)"""
    return [q['query'] for q in current_cypher_queries]
```

**Modify `PankBaseAgent/utils.py`** to mark final queries:

```python
def _pankbase_api_query_core(input: str, q: Queue) -> None:
    try:
        agent = _get_text2cypher_agent()
        
        # Use refinement with test-time scaling
        cypher_result = agent.respond(input)
        
        from .text_to_cypher.src.cypher_validator import validate_cypher
        validation = validate_cypher(cypher_result)
        
        # If score is low, use iterative refinement
        if validation['score'] < 95:
            refinement_result = agent.respond_with_refinement(input, max_iterations=5)
            cypher_result = refinement_result['cypher']
            # ... logging ...
        
        cleaned_cypher = clean_cypher_for_json(cypher_result)
        
        # ONLY add the final successful query
        add_cypher_query(cleaned_cypher, is_final=True)  # ← Changed
        
        # ... rest of function (execute query) ...
```

**Modify `main.py`** to use final queries:

```python
def chat_one_round(messages_history: list[dict], question: str) -> Tuple[list[dict], str]:
    reset_cypher_queries()
    # ... conversation loop ...
    
    if (response['to'] == 'user'):
        # Get only final successful queries
        cypher_queries = get_final_cypher_queries()  # ← Changed
        
        original_question = question.replace('====== From User ======\n', '')
        format_input = f"Human Query: {original_question}\n\nCypher Queries: {json.dumps(cypher_queries)}\n\nFinal Answer: {json.dumps(response['text'])}"
        format_result = format_agent(format_input)
        
        return (messages, format_result)
```

### Option 2: Only Add Final Queries (Simpler)

**Don't call `add_cypher_query()` during refinement** - only call it once with the final query.

**Modify `PankBaseAgent/utils.py`**:

```python
def _pankbase_api_query_core(input: str, q: Queue) -> None:
    try:
        agent = _get_text2cypher_agent()
        
        # Generate and refine (if needed)
        cypher_result = agent.respond(input)
        validation = validate_cypher(cypher_result)
        
        if validation['score'] < 95:
            refinement_result = agent.respond_with_refinement(input, max_iterations=5)
            cypher_result = refinement_result['cypher']
        
        cleaned_cypher = clean_cypher_for_json(cypher_result)
        
        # ONLY add the final query (after all refinement)
        add_cypher_query(cleaned_cypher)  # ← Only called once per query
        
        # Execute query
        session = _get_pankbase_session()
        response = session.post(...)
        # ...
```

**Remove any `add_cypher_query()` calls from `text2cypher_agent.py`** if they exist.

## Recommended Approach

**Use Option 2 (Simpler)** because:
1. ✅ Less code changes
2. ✅ Clearer intent - only final queries are tracked
3. ✅ No need to change data structure
4. ✅ Backward compatible

## Implementation Steps

### Step 1: Verify Current Calls

Check where `add_cypher_query()` is currently called:
```bash
grep -n "add_cypher_query" PankBaseAgent/utils.py
grep -n "add_cypher_query" PankBaseAgent/text_to_cypher/src/*.py
```

### Step 2: Ensure Single Call

Make sure `add_cypher_query()` is called ONLY ONCE per query, AFTER refinement:

```python
# In PankBaseAgent/utils.py _pankbase_api_query_core()

# Generate initial query
cypher_result = agent.respond(input)
validation = validate_cypher(cypher_result)

# Refine if needed (DON'T add to list yet)
if validation['score'] < 95:
    refinement_result = agent.respond_with_refinement(input, max_iterations=5)
    cypher_result = refinement_result['cypher']
    # Log refinement details...

# Clean and add ONLY the final query
cleaned_cypher = clean_cypher_for_json(cypher_result)
add_cypher_query(cleaned_cypher)  # ← Single call

# Execute the query
session = _get_pankbase_session()
response = session.post(...)
```

### Step 3: Update FormatAgent Prompt

**Modify `prompts/format_prompt.txt`**:

```markdown
5. **Cypher Queries**
   - Include **only the final successful** Cypher queries that contributed to the answer.
   - These are the queries that were actually executed and returned results.
   - Order them by **relevance** to the Human Query (most relevant first).
   - Do NOT include:
     - Failed queries from refinement iterations
     - Exploratory queries that didn't contribute
     - Duplicate queries
   - If none are provided, use an empty array `[]`.
   - Never use placeholders like "null" or "N/A."
```

### Step 4: Test

**Test Case 1: Simple Query (No Refinement)**
```
User: "Find gene CFTR"
→ 1 query generated, score 100
→ FormatAgent receives: 1 query ✅
```

**Test Case 2: Query with Refinement**
```
User: "Find upregulated genes in beta cells"
→ Iteration 1: score 60 (not added)
→ Iteration 2: score 100 (added)
→ FormatAgent receives: 1 query ✅
```

**Test Case 3: Multiple Sub-Queries**
```
User: "What QTLs link CFTR to T1D?"
PankBaseAgent decomposes:
  → Query 1: "Find gene CFTR" (final query added)
  → Query 2: "Get all QTL relationships" (final query added)
→ FormatAgent receives: 2 queries ✅
```

## Benefits

1. ✅ **Cleaner Output**: Only relevant queries shown to user
2. ✅ **Better UX**: User sees what actually happened, not internal iterations
3. ✅ **Easier Debugging**: Clear which queries were used
4. ✅ **Accurate Provenance**: Cypher queries match the actual answer

## Alternative: Filter in FormatAgent

If you don't want to change the tracking logic, you could filter in the FormatAgent prompt:

**Add to `format_prompt.txt`**:

```markdown
5. **Cypher Query Filtering**
   - You will receive ALL Cypher queries generated, including refinement iterations.
   - Your job: Identify and include ONLY the queries that contributed to the final answer.
   - Look for patterns:
     - Queries with correct syntax and format
     - Queries that match the entities mentioned in the answer
     - The LAST query in each refinement sequence
   - Exclude:
     - Queries with obvious errors (missing RETURN, wrong format)
     - Intermediate refinement attempts
     - Queries for entities not mentioned in the answer
```

**Pros**: No code changes needed
**Cons**: 
- ❌ Relies on LLM to filter correctly
- ❌ More token usage (sending all queries)
- ❌ Less reliable

## Recommendation

**Implement Option 2 (Only Add Final Queries)** because:
- Simple code change
- Deterministic (no LLM filtering)
- Efficient (less token usage)
- Clear semantics

The key change is ensuring `add_cypher_query()` is called ONLY ONCE per query, AFTER all refinement is complete.

