# WHERE Constraint Enforcement

## Problem

Text2Cypher was generating unconstrained queries that return ALL nodes:

```cypher
MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene)
WITH collect(DISTINCT sn) AS nodes, collect(r) AS edges
RETURN nodes, edges;
```

**Issues**:
- Returns **ALL SNPs** in the database (potentially thousands)
- No filtering by gene name, SNP name, or any property
- Causes performance problems and irrelevant results
- User likely wanted SNPs for a **specific gene**, not all SNPs

## Root Causes

1. **PlannerAgent** sending broad queries: "Get all SNPs that have part_of_QTL_signal relationships"
2. **Text2Cypher** not adding WHERE constraints even when entity names are mentioned
3. **No validation** for unconstrained queries

## Solution

### 1. Updated Text2Cypher System Prompt

**File**: `PankBaseAgent/text_to_cypher/src/text2cypher_agent.py`

**Added Rule #6** (lines 27-30):
```
6. ALWAYS use WHERE constraints to filter results (by name, id, or properties)
   - If query mentions specific entities (gene name, SNP name, etc.), add WHERE clause
   - Avoid unconstrained queries that return ALL nodes (e.g., MATCH (sn:snp) without WHERE)
   - Use properties like .name, .id, or relationship properties to filter
```

**Added Good Examples** (lines 32-46):
```cypher
Query: 'Find gene with name CFTR'
MATCH (g:gene) WHERE g.name = 'CFTR'
WITH collect(DISTINCT g) AS nodes, [] AS edges
RETURN nodes, edges;

Query: 'Get SNPs that have part_of_QTL_signal relationships with gene CFTR'
MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene) WHERE g.name = 'CFTR'
WITH collect(DISTINCT sn)+collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;

Query: 'Get upregulated genes in Beta Cell'
MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type) WHERE ct.name = 'Beta Cell' AND deg.UpOrDownRegulation = 'up'
WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges
RETURN nodes, edges;
```

**Added Bad Examples** (lines 48-51):
```
WRONG: MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene) (no WHERE - returns ALL SNPs!)
WRONG: MATCH (g:gene)-[:function_annotation]->(fo:gene_ontology) (missing variable name)
WRONG: MATCH (g:gene) (no WHERE - returns ALL genes!)
```

### 2. Added WHERE Constraint Validator

**File**: `PankBaseAgent/text_to_cypher/src/cypher_validator.py`

**New Function** (lines 256-293):
```python
def check_where_constraints(cypher: str) -> List[str]:
    """
    Check that queries have WHERE constraints to avoid returning ALL nodes.
    
    Warns if MATCH has no WHERE clause (likely returns too many results).
    """
    warnings = []
    
    # Find MATCH clauses
    match_positions = [m.start() for m in re.finditer(r'\bMATCH\b', cypher_upper)]
    
    for match_pos in match_positions:
        # Find the next WITH, RETURN, or MATCH after this MATCH
        rest_of_query = cypher_upper[match_pos:]
        
        # Look for WHERE before the next clause
        next_clause_match = re.search(r'\b(WITH|RETURN|MATCH)\b', rest_of_query[6:])
        
        if next_clause_match:
            segment = rest_of_query[:next_clause_match.start() + 6]
        else:
            segment = rest_of_query
        
        # Check if this segment has WHERE
        if 'WHERE' not in segment:
            warnings.append(
                f"Query may return too many results: MATCH without WHERE constraint. "
                f"Consider adding WHERE clause to filter by name, id, or properties."
            )
    
    return warnings
```

**Integrated as Check 5.5** (lines 86-91):
```python
# Check 5.5: WHERE constraints (IMPORTANT - warning only)
where_warnings = check_where_constraints(cypher)
if where_warnings:
    warnings.extend(where_warnings)
else:
    passed_checks.append("Query has appropriate WHERE constraints")
```

**Note**: This is a **warning**, not an error (doesn't deduct points), but appears in refinement feedback.

### 3. Updated PankBaseAgent Prompt

**File**: `PankBaseAgent/prompts/general_prompt.txt`

**Updated Example** (lines 54-66):
```markdown
**Good decomposition** (7 atomic queries in parallel):
1. "Find SNP with name rs2402203"
2. "Find gene with name CFTR"
3. "Find disease with name type 1 diabetes"
4. "Get SNPs that have part_of_QTL_signal relationships with gene CFTR"  ← Specific gene
5. "Get genes that are effector genes for disease type 1 diabetes"  ← Specific disease
6. "Get genes that have DEG_in relationships with cell type Beta Cell"  ← Specific cell type
7. "Get cell types with name Beta Cell"

**IMPORTANT**: Always specify entity names/IDs in queries (e.g., "gene CFTR", "SNP rs2402203", "cell type Beta Cell").
Avoid broad queries like "Get all SNPs" or "Get all genes" which return too much data.
```

### 4. Updated PlannerAgent Prompt

**File**: `prompts/general_prompt.txt`

**Updated Strategy** (lines 22-34):
```markdown
**CRITICAL DECOMPOSITION STRATEGY**: Break complex questions into **many simple, atomic sub-queries** (typically 3-10 queries).
- Each sub-query should be **extremely simple** - one entity lookup or one relationship traversal
- **ALWAYS specify entity names/IDs** in queries (e.g., "gene CFTR", "SNP rs2402203", "cell type Beta Cell")
- **AVOID broad queries** like "Get all SNPs" or "Get all genes" (returns too much data)
- The aim is to explore relationships among key terms by querying each component separately
- Example: "How does SNP rs2402203 contribute to CFTR's function in T1D?"
  - Key terms: rs2402203, CFTR, T1D
  - ✅ GOOD: "Find SNP rs2402203" → "Find gene CFTR" → "Get SNPs with part_of_QTL_signal relationships to gene CFTR"
  - ❌ BAD: "Get all SNPs" → "Get all genes" → "Get all QTL relationships"
- **All sub-queries execute in parallel**, so more queries ≠ slower (up to reasonable limits)
- You synthesize all results in your final answer

**Prefer many simple, specific queries over few broad ones** - the Text2Cypher agent performs better with constrained operations
```

## Examples

### Before (Bad)

**PlannerAgent sends**:
```
"Get all SNPs that have part_of_QTL_signal relationships with genes"
```

**Text2Cypher generates**:
```cypher
MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene)
WITH collect(DISTINCT sn) AS nodes, collect(r) AS edges
RETURN nodes, edges;
```

**Result**: Returns 10,000+ SNPs (too much data!)

### After (Good)

**PlannerAgent sends**:
```
"Get SNPs that have part_of_QTL_signal relationships with gene CFTR"
```

**Text2Cypher generates**:
```cypher
MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene)
WHERE g.name = 'CFTR'
WITH collect(DISTINCT sn)+collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;
```

**Result**: Returns only SNPs related to CFTR (relevant data!)

## Validation Examples

### Test 1: Unconstrained Query (Warning)

**Query**:
```cypher
MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene)
WITH collect(DISTINCT sn)+collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;
```

**Validation Result**:
```
Score: 95/100

WARNINGS (should fix):
  1. Query may return too many results: MATCH without WHERE constraint.
     Consider adding WHERE clause to filter by name, id, or properties.
```

### Test 2: Constrained Query (Pass)

**Query**:
```cypher
MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene)
WHERE g.name = 'CFTR'
WITH collect(DISTINCT sn)+collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;
```

**Validation Result**:
```
Score: 100/100

PASSED CHECKS:
  ✓ Query has appropriate WHERE constraints
  ✓ All relationships have variable names
  ✓ Correct return format with nodes and edges
  ✓ All collect() use DISTINCT
```

## When WHERE is Required

### Always Use WHERE For:

1. **Entity Lookups by Name**:
   ```cypher
   MATCH (g:gene) WHERE g.name = 'CFTR'
   ```

2. **Entity Lookups by ID**:
   ```cypher
   MATCH (g:gene) WHERE g.id = 'ENSG00000001626'
   ```

3. **Relationship Queries with Specific Entities**:
   ```cypher
   MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene) WHERE g.name = 'CFTR'
   ```

4. **Property Filtering**:
   ```cypher
   MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type)
   WHERE ct.name = 'Beta Cell' AND deg.UpOrDownRegulation = 'up'
   ```

### Exceptions (WHERE Optional):

1. **Counting All Nodes** (if explicitly requested):
   ```cypher
   MATCH (g:gene) RETURN count(g)
   ```

2. **Schema Exploration** (if explicitly requested):
   ```cypher
   MATCH (ct:cell_type) RETURN DISTINCT ct.name
   ```

But these should be **rare** - most queries should filter!

## Benefits

1. **Performance**: Queries return only relevant data (not thousands of nodes)
2. **Accuracy**: Results match user intent (specific entities, not all entities)
3. **Refinement**: LLM sees warning and adds WHERE in next iteration
4. **Guidance**: Clear examples teach Text2Cypher to use WHERE

## Testing

Run the test suite:

```bash
cd PankBaseAgent/text_to_cypher
python3 test_where_constraints.py
```

**Expected Results**:
- Unconstrained queries: Get warnings
- Constrained queries: Score 100/100

## Monitoring

Check logs for:

1. **Unconstrained queries**:
   ```
   DEBUG: Validation warnings: Query may return too many results
   ```

2. **Refinement fixing it**:
   ```
   Iteration 1: score=95 (warning about WHERE)
   Iteration 2: score=100 (added WHERE g.name = 'CFTR')
   ```

## Summary

✅ **Changes complete!**

The system now:
- Teaches Text2Cypher to always use WHERE constraints
- Validates queries for missing WHERE clauses
- Provides clear examples of constrained vs unconstrained queries
- Guides PlannerAgent to specify entity names in sub-queries
- Warns during refinement if WHERE is missing

**Key principle**: Every query should filter by name, id, or properties - avoid returning ALL nodes!

