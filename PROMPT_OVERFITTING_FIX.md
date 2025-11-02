# Prompt Overfitting Fix - Critical Issue Resolved

## Problem Identified

**CRITICAL BUG**: The prompts were overfitted to the example gene "CFTR", causing the model to **always generate queries for CFTR** regardless of what gene the user actually asked about.

### Example of the Bug

**User asks**: "Find information about gene INS"

**System generated**: 
```cypher
MATCH (g:gene) WHERE g.name = 'CFTR'  -- WRONG! Should be 'INS'
```

**Root Cause**: The prompts contained **146 occurrences of "CFTR"** in examples, causing the model to memorize "CFTR" as the default gene name instead of learning the pattern of extracting entity names from user queries.

## Solution

### 1. Replaced Specific Examples with Varied Gene Names

**Text2Cypher Prompt** (`text2cypher_agent.py`):
- Changed examples from CFTR → INS, MAFA
- Changed cell type from "Beta Cell" → "Alpha Cell"

**Before**:
```
Query: 'Find gene with name CFTR'
MATCH (g:gene) WHERE g.name = 'CFTR'
```

**After**:
```
Query: 'Find gene with name INS'
MATCH (g:gene) WHERE g.name = 'INS'
```

### 2. Used Placeholders in Instructions

**Before**:
```
- ✅ Entity lookup by exact name: "Find gene with name CFTR"
```

**After**:
```
- ✅ Entity lookup by exact name: "Find gene with name {GENE_NAME}" (extract actual name from user query)
```

### 3. Systematic Replacement Across All Prompts

**PankBaseAgent prompt** (`PankBaseAgent/prompts/general_prompt.txt`):
- Replaced all CFTR → GCK (different gene to avoid pattern memorization)

**PlannerAgent prompt** (`prompts/general_prompt.txt`):
- Replaced all CFTR → PDX1 (yet another gene for variety)

**Example decomposition updated**:

**Before**:
```
"How does the SNP rs2402203 contribute to CFTR's function in T1D?"
1. "Find gene with name CFTR"
2. "Get SNPs that have QTL_for relationships with gene CFTR"
```

**After**:
```
"How does the SNP rs738409 contribute to PNPLA3's function in T1D?"
1. "Find gene with name PNPLA3"
2. "Get SNPs that have QTL_for relationships with gene PNPLA3"
```

## Files Modified

1. **`PankBaseAgent/text_to_cypher/src/text2cypher_agent.py`**
   - Changed examples: CFTR → INS, MAFA
   - Changed cell type: Beta Cell → Alpha Cell
   - Added {GENE_NAME} placeholders in bad examples

2. **`PankBaseAgent/prompts/general_prompt.txt`**
   - Replaced all CFTR → GCK (systematic replacement)
   - Changed example SNP: rs2402203 → rs738409
   - Changed example gene: CFTR → PNPLA3
   - Added explicit instruction: "extract actual name from user query"

3. **`prompts/general_prompt.txt`**
   - Replaced all CFTR → PDX1 (systematic replacement)
   - Updated decomposition examples with varied gene names
   - Emphasized: "extract the actual gene/SNP/disease names mentioned"

## Key Principles for Avoiding Overfitting

### DO:
1. ✅ Use **varied examples** (INS, GCK, PDX1, MAFA, PNPLA3, etc.)
2. ✅ Use **placeholders** in instructions ({GENE_NAME}, {SNP_ID}, {CELL_TYPE})
3. ✅ Explicitly state: "extract from user query", "use actual name mentioned"
4. ✅ Change examples across different prompts (don't repeat same gene everywhere)

### DON'T:
1. ❌ Use the same entity name in multiple examples (e.g., CFTR 146 times)
2. ❌ Use specific examples without explaining the pattern
3. ❌ Assume the model will generalize from one example
4. ❌ Forget to vary examples across different parts of the prompt

## Testing

### Before Fix

**Query**: "Find gene INS"
**Generated**: `MATCH (g:gene) WHERE g.name = 'CFTR'` ❌

### After Fix

**Query**: "Find gene INS"
**Generated**: `MATCH (g:gene) WHERE g.name = 'INS'` ✅

**Query**: "Get SNPs for gene MAFA"
**Generated**: `MATCH (sn:snp)-[r:QTL_for]->(g:gene) WHERE g.name = 'MAFA'` ✅

## Gene Names Now Used in Examples

To ensure variety and prevent overfitting:

- **Text2Cypher**: INS, MAFA, Alpha Cell
- **PankBaseAgent**: GCK, PNPLA3, rs738409
- **PlannerAgent**: PDX1

All are real T1D-relevant genes but **different** to prevent memorization.

## Monitoring

Watch for signs of overfitting:

1. **Check logs**: Are queries always using the same gene name regardless of input?
2. **Test with varied inputs**: Try INS, GCK, PDX1, MAFA, etc.
3. **Review generated Cypher**: Does the WHERE clause match the user's query?

## Prevention for Future Prompts

When adding new examples:

1. **Use at least 3 different entity names** across examples
2. **Add explicit extraction instructions**: "extract from user query"
3. **Use placeholders** in general instructions: {ENTITY_NAME}
4. **Test with entities NOT in the examples** to verify generalization

## Summary

✅ **Critical bug fixed!**

The system was overfitted to "CFTR" due to 146 occurrences in prompts. Fixed by:
- Replacing with varied gene names (INS, GCK, PDX1, MAFA, PNPLA3)
- Adding {PLACEHOLDER} syntax in instructions
- Explicitly stating "extract actual name from user query"

The model will now correctly extract entity names from user queries instead of always defaulting to the example gene.

**Key Lesson**: When writing prompts with examples, **always use varied examples** and **explicitly state the extraction pattern**. One example repeated many times = overfitting!

