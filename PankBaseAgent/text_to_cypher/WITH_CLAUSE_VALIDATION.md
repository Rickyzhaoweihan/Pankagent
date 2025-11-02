# WITH Clause Validation - Custom Regex-Based Solution

## Summary

Implemented robust **custom regex-based validation** for Cypher WITH clauses. No external dependencies (CyVer) needed!

## The Problem You Reported

Bad query:
```cypher
MATCH (sn:snp)-[r:QTL_for]->(g:gene) 
WITH DISTINCT sn, r AS nodes, edges 
RETURN nodes, edges;
```

**Issues**:
1. `DISTINCT` outside `collect()`
2. Missing `collect()` for variables
3. Ambiguous syntax

## The Solution

### New Function: `check_with_clause_structure()`

**Location**: `cypher_validator.py` lines 169-250

**What it catches**:
1. ✅ `DISTINCT` outside `collect()` functions
2. ✅ Missing `collect()` for nodes assignment
3. ✅ Missing `collect()` for edges assignment
4. ✅ Missing nodes or edges variables
5. ✅ Inconsistent collect usage

### Key Regex Patterns

**For nodes**:
```python
r'((?:collect\([^)]+\)|\w+)(?:\s*\+\s*(?:collect\([^)]+\)|\w+))*)\s+AS\s+nodes'
```

This matches:
- ✅ `collect(DISTINCT sn) AS nodes` (valid)
- ✅ `collect(DISTINCT sn)+collect(DISTINCT g) AS nodes` (valid)
- ❌ `sn AS nodes` (invalid - no collect)
- ❌ `r AS nodes` (invalid - no collect)

**For edges**:
```python
r'((?:collect\([^)]+\)|\w+|\[\])(?:\s*\+\s*(?:collect\([^)]+\)|\w+))*)\s+AS\s+edges'
```

This matches:
- ✅ `collect(DISTINCT r) AS edges` (valid)
- ✅ `[] AS edges` (valid - empty edges)
- ❌ `r AS edges` (invalid - no collect)

## Test Results

### Test 1: Your Bad Query
```cypher
MATCH (sn:snp)-[r:QTL_for]->(g:gene) 
WITH DISTINCT sn, r AS nodes, edges 
RETURN nodes, edges;
```

**Score**: 25/100 ✅

**Errors Caught**:
1. DISTINCT must be inside collect() functions
2. WITH clause must define 'edges' variable
3. Variables assigned to 'nodes' must use collect()
4. Wrong return format

### Test 2: Missing collect()
```cypher
MATCH (sn:snp)-[r:QTL_for]->(g:gene)
WITH sn AS nodes, r AS edges
RETURN nodes, edges;
```

**Score**: 50/100 ✅

**Errors Caught**:
1. Variables assigned to 'nodes' must use collect()
2. Variables assigned to 'edges' must use collect()

### Test 3: Correct Query
```cypher
MATCH (sn:snp)-[r:QTL_for]->(g:gene)
WITH collect(DISTINCT sn)+collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;
```

**Score**: 100/100 ✅

**All checks passed!**

## Integration

**Check #1** in `validate_cypher()` (CRITICAL - 35 points)

```python
# Check 1: WITH clause structure (CRITICAL - 35 points)
with_errors = check_with_clause_structure(cypher)
if with_errors:
    errors.extend(with_errors)
else:
    passed_checks.append("WITH clause properly structured with collect()")
```

## Scoring

- **WITH clause errors**: -35 points (CRITICAL)
- **Relationship naming errors**: -30 points
- **Return format errors**: -25 points
- **DISTINCT errors**: -15 points
- **Disease naming errors**: -15 points
- **Property value errors**: -20 points
- **Relationship direction errors**: -25 points

## Why Not CyVer?

**CyVer Issues**:
- ❌ Requires Neo4j driver connection
- ❌ You're using HTTP API, not direct driver
- ❌ Extra dependency
- ❌ More complex setup

**Custom Regex Benefits**:
- ✅ No external dependencies
- ✅ Fast and deterministic
- ✅ Works offline (no database needed)
- ✅ Catches the exact errors you need
- ✅ Easy to maintain and extend

## Files Modified

1. **`cypher_validator.py`**:
   - Removed all CyVer code
   - Added `check_with_clause_structure()` function
   - Fixed regex patterns to properly match collect() expressions
   - Integrated as Check #1 (35 points)

2. **Deleted**:
   - `test_cyver_api.py` (not needed)
   - `test_cyver_integration.py` (not needed)
   - `CYVER_INTEGRATION.md` (obsolete)

3. **Kept**:
   - `test_with_clause_validation.py` (tests custom validation)

## Usage

### Run Tests
```bash
cd PankBaseAgent/text_to_cypher
python3 test_with_clause_validation.py
```

### Use in Code
```python
from cypher_validator import validate_cypher

result = validate_cypher(your_cypher_query)

if result['score'] < 90:
    print("Errors found:")
    for error in result['errors']:
        print(f"  - {error}")
```

### In Refinement Loop
```python
# In text2cypher_agent.py
validation = validate_cypher(cypher)

if validation['score'] < 90:
    # Build refinement prompt with errors
    feedback = format_validation_report(validation)
    # LLM sees clear error messages and can fix them
```

## Example Error Messages

**Clear and actionable**:

1. `DISTINCT must be inside collect() functions. Use: WITH collect(DISTINCT var) AS nodes, ... Not: WITH DISTINCT var, ...`

2. `Variables assigned to 'nodes' must use collect(). Found: sn AS nodes. Should be: collect(DISTINCT ...) AS nodes`

3. `Variables assigned to 'edges' must use collect() or []. Found: r AS edges. Should be: collect(DISTINCT ...) AS edges`

4. `WITH clause must define 'edges' variable (... AS edges)`

## Conclusion

✅ **Problem solved without CyVer!**

The custom regex-based validation:
- Catches your exact error case
- Works with your HTTP API architecture
- No external dependencies
- Fast and reliable
- Easy to maintain

**Your bad query now scores 25/100** (correctly flagged as bad)
**Your good query now scores 100/100** (correctly validated as good)

Perfect for the refinement loop! 🎉

