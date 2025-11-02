# Query Decomposition Strategy for PankBaseAgent

## Problem Analysis

### Current Issues

Looking at `log.txt`, the Text2CypherAgent struggles with complex queries like:
- "Retrieve differential expression data for CFTR in T1D vs non-diabetic samples"
  - This requires understanding: gene lookup, cell type context, expression comparison, disease filtering
  - **Too complex** for a 9B model with 8k context

### Root Cause

The PankBaseAgent's current prompt (line 77 in `general_prompt.txt`):
> "Each function call should query **only one biological relation or node type**"

This is **not specific enough**. It allows queries like:
- ❌ "Find upregulated genes in beta cells with log2FC > 2 and p-value < 0.05"
- ❌ "Get QTL associations linking CFTR to T1D with high PIP scores"

These are still **multi-step queries** that require:
1. Entity lookup (gene/SNP by name)
2. Relationship traversal
3. Property filtering
4. Aggregation/comparison

## Solution: Atomic Query Decomposition

### Principle: One Operation Per Query

Each sub-query should perform **exactly ONE operation**:

1. **Entity Lookup**: Find a specific node by name/ID
2. **Direct Relationship**: Get immediate neighbors via one relationship type
3. **Property Retrieval**: Get properties of known nodes
4. **Simple Filter**: Filter by ONE property condition

### Query Complexity Levels

```
Level 1: SIMPLE (Text2Cypher can handle)
├─ "Find gene with name CFTR"
├─ "Get all genes that are effector genes for type 1 diabetes"
├─ "Find cell type with name Beta Cell"
└─ "Get SNP with ID rs738409"

Level 2: MODERATE (Text2Cypher struggles)
├─ "Find upregulated genes in beta cells"
│   └─ Requires: gene lookup + DEG_in relationship + cell type filter + regulation filter
├─ "Get QTL associations for CFTR"
│   └─ Requires: gene lookup + QTL_for relationship + property retrieval
└─ "Find genes with high expression in beta cells"
    └─ Requires: cell type lookup + expression_level_in relationship + threshold filter

Level 3: COMPLEX (Text2Cypher fails)
├─ "Retrieve differential expression data for CFTR in T1D vs non-diabetic samples"
│   └─ Requires: gene lookup + DEG_in + cell type context + expression comparison + disease filtering
├─ "Find QTL associations linking CFTR to T1D with PIP > 0.5"
│   └─ Requires: gene lookup + disease context + QTL_for + PIP filtering + credible set analysis
└─ "Get genes upregulated in beta cells with log2FC > 2 and adjusted p-value < 0.05"
    └─ Requires: cell type lookup + DEG_in + multiple property filters + statistical thresholds
```

### Decomposition Strategy

**Complex Query Example:**
"Find QTL associations linking CFTR to T1D with high confidence"

**Current Decomposition** (PankBaseAgent):
```python
functions = [
  {"name": "pankbase_api_query", "input": "Find QTL associations for gene CFTR"},
  {"name": "pankbase_api_query", "input": "Get effector genes for type 1 diabetes"}
]
```
❌ **Problem**: First query is still too complex (gene lookup + QTL traversal)

**Improved Decomposition** (Atomic):
```python
functions = [
  {"name": "pankbase_api_query", "input": "Find gene with name CFTR"},
  {"name": "pankbase_api_query", "input": "Get all SNPs that have QTL associations with any gene"},
  {"name": "pankbase_api_query", "input": "Find disease with name type 1 diabetes"},
  {"name": "pankbase_api_query", "input": "Get all genes that are effector genes for type 1 diabetes"}
]
```
✅ **Better**: Each query is atomic, Text2Cypher can handle them

**Post-Processing** (PankBaseAgent):
- Filter QTL results for CFTR
- Cross-reference with T1D effector genes
- Apply PIP threshold
- Synthesize answer

## Implementation Plan

### 1. Update PankBaseAgent Prompt

**File**: `prompts/general_prompt.txt`

**Current** (lines 76-78):
```
1. All function calls must be **read-only** — no modification or deletion.  
2. Each function call should query **only one biological relation or node type**.  
3. Independent queries should be executed **in parallel**.
```

**New** (replace with):
```
1. All function calls must be **read-only** — no modification or deletion.  
2. **CRITICAL**: Each function call must be **ATOMIC** - perform only ONE simple operation:
   - Entity lookup by name/ID (e.g., "Find gene with name CFTR")
   - Direct relationship traversal (e.g., "Get all genes that are effector genes for type 1 diabetes")
   - Property retrieval for known entities
   - Simple single-property filter (e.g., "Get genes with UpOrDownRegulation equal to up")
3. **AVOID COMPLEX QUERIES**: Do NOT combine multiple operations in one query:
   - ❌ BAD: "Find upregulated genes in beta cells with log2FC > 2"
   - ✅ GOOD: "Find genes that have DEG_in relationships with Beta Cell cell type"
   - ✅ GOOD: "Get all cell types with name Beta Cell"
4. Break complex questions into 2-3 **simple, atomic sub-queries**.
5. Independent queries should be executed **in parallel**.
6. You will **synthesize and filter** the results in your final answer.
```

### 2. Add Query Decomposition Examples

**Add to prompt** (after line 161):

```markdown
## Query Decomposition Examples

### Example 1: Complex Query Decomposition

**User asks**: "Find upregulated genes in beta cells for T1D"

**❌ BAD Decomposition** (too complex):
```json
{
  "functions": [
    {"name": "pankbase_api_query", "input": "Find upregulated genes in beta cells"}
  ]
}
```

**✅ GOOD Decomposition** (atomic):
```json
{
  "functions": [
    {"name": "pankbase_api_query", "input": "Find cell type with name Beta Cell"},
    {"name": "pankbase_api_query", "input": "Get all genes that have DEG_in relationships"},
    {"name": "pankbase_api_query", "input": "Find disease with name type 1 diabetes"}
  ]
}
```
Then in your final answer, filter for genes with UpOrDownRegulation='up' in Beta Cell context.

### Example 2: QTL Query Decomposition

**User asks**: "What QTLs are associated with CFTR?"

**❌ BAD Decomposition**:
```json
{
  "functions": [
    {"name": "pankbase_api_query", "input": "Find QTL associations for gene CFTR with high PIP"}
  ]
}
```

**✅ GOOD Decomposition**:
```json
{
  "functions": [
    {"name": "pankbase_api_query", "input": "Find gene with name CFTR"},
    {"name": "pankbase_api_query", "input": "Get all SNPs that have QTL_for relationships with genes"}
  ]
}
```
Then filter for CFTR-related QTLs in your synthesis.

### Example 3: Gene Expression Query

**User asks**: "Is CFTR differentially expressed in T1D?"

**❌ BAD Decomposition**:
```json
{
  "functions": [
    {"name": "pankbase_api_query", "input": "Retrieve differential expression data for CFTR in T1D vs non-diabetic samples"}
  ]
}
```

**✅ GOOD Decomposition**:
```json
{
  "functions": [
    {"name": "pankbase_api_query", "input": "Find gene with name CFTR"},
    {"name": "pankbase_api_query", "input": "Get all genes that have DEG_in relationships with cell types"},
    {"name": "pankbase_api_query", "input": "Get all genes that are effector genes for type 1 diabetes"}
  ]
}
```
Then synthesize: Check if CFTR appears in DEG results and cross-reference with T1D effector genes.

## Key Principles

1. **One Hop Maximum**: Each query should traverse at most ONE relationship type
2. **No Complex Filters**: Avoid combining multiple WHERE conditions
3. **Name-Based Lookup**: Always use exact entity names (e.g., "Beta Cell", "type 1 diabetes")
4. **Post-Processing**: Let the PankBaseAgent filter and synthesize results
5. **Parallel Execution**: Run independent atomic queries in parallel
```

### 3. Add Query Validation Helper

**New function in `utils.py`**:

```python
def validate_query_complexity(query: str) -> tuple[bool, str]:
    """
    Validate that a query is atomic enough for Text2CypherAgent.
    
    Returns:
        (is_valid, reason)
    """
    query_lower = query.lower()
    
    # Check for multiple operations
    complexity_indicators = [
        ('with.*and', 'Multiple AND conditions'),
        ('log2fc.*p.?value', 'Multiple property filters'),
        ('upregulated.*in.*for', 'Too many context requirements'),
        ('differential.*expression.*vs', 'Comparison operations'),
        ('high.*low|greater.*less', 'Relative comparisons'),
        ('linking.*to.*with', 'Multi-entity relationships'),
    ]
    
    for pattern, reason in complexity_indicators:
        if re.search(pattern, query_lower):
            return (False, f"Query too complex: {reason}. Break into atomic sub-queries.")
    
    # Check for good patterns
    good_patterns = [
        r'find \w+ with name',
        r'get all \w+ that (have|are)',
        r'retrieve \w+ for \w+$',
    ]
    
    for pattern in good_patterns:
        if re.search(pattern, query_lower):
            return (True, "Query is atomic")
    
    # Default: warn if query is long
    if len(query.split()) > 12:
        return (False, "Query may be too complex (>12 words). Consider breaking it down.")
    
    return (True, "Query appears atomic")
```

### 4. Add Query Complexity Logging

**Modify `_pankbase_api_query_core` in `utils.py`**:

```python
def _pankbase_api_query_core(input: str, q: Queue) -> None:
    try:
        # Log query complexity
        is_valid, reason = validate_query_complexity(input)
        if not is_valid:
            print(f"WARNING: {reason}")
            with open('log.txt', 'a') as log_file:
                log_file.write(f"COMPLEXITY WARNING: {input}\n")
                log_file.write(f"  Reason: {reason}\n\n")
        
        agent = _get_text2cypher_agent()
        # ... rest of function
```

## Expected Improvements

### Before (Complex Queries)

**User**: "Find upregulated genes in beta cells"
**PankBaseAgent sends**: "Find upregulated genes in beta cells"
**Text2Cypher generates**:
```cypher
MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type)
WHERE ct.name='beta cell' AND deg.UpOrDownRegulation='upregulated'
...
```
**Score**: 60/100 (invalid values, wrong case)
**Refinement**: 5 iterations, final score 55/100 (still fails)

### After (Atomic Queries)

**User**: "Find upregulated genes in beta cells"
**PankBaseAgent sends**:
1. "Find cell type with name Beta Cell"
2. "Get all genes that have DEG_in relationships"

**Text2Cypher generates**:
```cypher
# Query 1
MATCH (ct:cell_type)
WHERE ct.name='Beta Cell'
WITH collect(DISTINCT ct) AS nodes, [] AS edges
RETURN nodes, edges;

# Query 2
MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type)
WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges
RETURN nodes, edges;
```
**Score**: 100/100 for both (no refinement needed)
**PankBaseAgent**: Filters results for UpOrDownRegulation='up' in Beta Cell

## Benefits

1. **Higher Success Rate**: Simple queries score 95-100/100 consistently
2. **Fewer Refinement Iterations**: Atomic queries rarely need refinement
3. **Better Token Efficiency**: Simple queries use less context
4. **Clearer Intent**: Each query has one clear purpose
5. **Easier Debugging**: Can identify which atomic query failed
6. **Scalable**: Works even as schema grows

## Trade-offs

1. **More API Calls**: 2-3 atomic queries instead of 1 complex query
2. **Post-Processing Burden**: PankBaseAgent must filter/synthesize results
3. **Potential Data Transfer**: May retrieve more data than needed
4. **Coordination Complexity**: Must align results from multiple queries

## Mitigation Strategies

1. **Parallel Execution**: Already supported (MAX_ITER=3)
2. **Smart Caching**: Cache common entity lookups (e.g., "Beta Cell", "type 1 diabetes")
3. **Result Limiting**: Add LIMIT clauses to atomic queries
4. **Intelligent Synthesis**: PankBaseAgent does smart filtering/joining

## Files to Modify

1. **`prompts/general_prompt.txt`** (+50 lines)
   - Update behavior rules with atomic query requirements
   - Add query decomposition examples
   - Add complexity guidelines

2. **`utils.py`** (+30 lines)
   - Add `validate_query_complexity()` function
   - Add complexity logging to `_pankbase_api_query_core`

3. **`claude.py`** (no changes)
   - System already supports multiple parallel function calls

## Testing Strategy

### Test Cases

1. **Simple Entity Lookup**
   - Input: "Find gene with name CFTR"
   - Expected: Single atomic query, score 100/100

2. **Direct Relationship**
   - Input: "Get all genes that are effector genes for type 1 diabetes"
   - Expected: Single atomic query, score 100/100

3. **Complex Query (Should Be Decomposed)**
   - Input: "Find upregulated genes in beta cells with log2FC > 2"
   - Expected: PankBaseAgent breaks into 2-3 atomic queries
   - Each atomic query scores 95-100/100

4. **Multi-Entity Query**
   - Input: "What QTLs link CFTR to T1D?"
   - Expected: 3 atomic queries (CFTR lookup, QTL retrieval, T1D lookup)
   - PankBaseAgent synthesizes results

### Success Metrics

- **Atomic Query Success Rate**: >95% score ≥90/100 on first attempt
- **Refinement Rate**: <10% of atomic queries need refinement
- **Overall Success**: >90% of user questions answered correctly
- **Token Efficiency**: <2000 tokens per atomic query (vs 4000+ for complex)

## Implementation Priority

1. **High Priority**: Update `general_prompt.txt` with atomic query rules
2. **Medium Priority**: Add query decomposition examples
3. **Low Priority**: Add complexity validation and logging

## Next Steps

1. Update the prompt with atomic query guidelines
2. Test with existing complex queries from log.txt
3. Monitor refinement rates and success scores
4. Iterate on decomposition examples based on real usage

