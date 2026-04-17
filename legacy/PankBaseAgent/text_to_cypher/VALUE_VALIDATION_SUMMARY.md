# Property Value Validation Feature

## Overview

The Cypher validator now checks property values against constrained valid values defined in `valid_property_values.json`. This prevents the LLM from hallucinating invalid values that would cause queries to return empty results.

## What Was Added

### 1. New Validation Function: `check_property_value_validity()`

**Location**: `src/cypher_validator.py`

**What it does**:
- Extracts property value comparisons from Cypher queries (e.g., `ct.name='beta cell'`)
- Checks if the value matches the valid values defined in `valid_property_values.json`
- Returns detailed error messages with:
  - The invalid value used
  - List of valid values
  - Helpful notes (e.g., case-sensitivity requirements)

**Example Error Message**:
```
Invalid value 'beta cell' for cell_type.name. Valid values: ['Acinar Cell', 'Alpha Cell', 'Beta Cell', 'Delta Cell', 'Ductal Cell', 'Endothelial Cell', 'Macrophage Cell', 'Stellate Cell']. Note: Case-sensitive. Use exact spelling as shown.
```

### 2. Integration into Main Validator

**Changes to `validate_cypher()`**:
- Added Check 7: Property value validity (CRITICAL - 20 points)
- Deducts 20 points for invalid property values
- Adds "All property values are valid" to passed checks when successful

### 3. Scoring Impact

**Score Deductions**:
- Invalid property values: **-20 points** (CRITICAL)
- This is a critical error because wrong values will cause queries to return no results

**Example Scores**:
```
Query with ct.name='beta cell':        Score: 80/100 (-20 for invalid value)
Query with ct.name='Beta Cell':        Score: 100/100 (valid value)
Query with deg.UpOrDownRegulation='upregulated': Score: 80/100 (-20 for invalid value)
Query with deg.UpOrDownRegulation='up':          Score: 100/100 (valid value)
```

## How It Works

### Step 1: LLM Generates Query with Invalid Value
```cypher
MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type)
WHERE ct.name='beta cell' AND deg.UpOrDownRegulation='upregulated'
WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges
RETURN nodes, edges;
```

### Step 2: Validator Detects Invalid Values
```
Validation Score: 60/100

ERRORS (must fix):
  1. Invalid value 'beta cell' for cell_type.name. Valid values: ['Acinar Cell', 'Alpha Cell', 'Beta Cell', 'Delta Cell', 'Ductal Cell', 'Endothelial Cell', 'Macrophage Cell', 'Stellate Cell']. Note: Case-sensitive. Use exact spelling as shown.
  2. Invalid value 'upregulated' for DEG_in.UpOrDownRegulation. Valid values: ['up', 'down']. Note: Lowercase only. 'up' means upregulated, 'down' means downregulated.
```

### Step 3: Refinement Prompt Includes Valid Values
The refinement prompt now includes:
1. **Validation errors** with exact valid values
2. **Detailed properties** for entities used in the query (from dynamic enrichment)
3. **Valid values** embedded in the property descriptions

### Step 4: LLM Fixes Values
```cypher
MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type)
WHERE ct.name='Beta Cell' AND deg.UpOrDownRegulation='up'
WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges
RETURN nodes, edges;
```

**New Score**: 100/100 ✅

## Currently Validated Properties

### Node Properties

#### `cell_type.name`
- **Valid Values**: 
  - "Acinar Cell"
  - "Alpha Cell"
  - "Beta Cell"
  - "Delta Cell"
  - "Ductal Cell"
  - "Endothelial Cell"
  - "Macrophage Cell"
  - "Stellate Cell"
- **Note**: Case-sensitive. Use exact spelling as shown.

#### `disease.name`
- **Valid Values**: 
  - "type 1 diabetes"
- **Note**: MUST be lowercase. Do NOT use 'T1D', 'Type 1 Diabetes', or any other variant.

### Relationship Properties

#### `DEG_in.UpOrDownRegulation`
- **Valid Values**: 
  - "up"
  - "down"
- **Note**: Lowercase only. 'up' means upregulated, 'down' means downregulated.

## Benefits

### 1. Prevents Empty Results
Invalid values cause queries to return no results. This validation catches them before execution.

**Before**:
```cypher
WHERE ct.name='beta cell'  -- Returns 0 results (wrong case)
```

**After**:
```cypher
WHERE ct.name='Beta Cell'  -- Returns actual results
```

### 2. Clear Error Messages
The LLM receives exact valid values and helpful notes during refinement.

### 3. Automatic Correction
The refinement loop uses these errors to automatically fix invalid values.

### 4. Extensible
Easy to add new constrained properties by editing `valid_property_values.json`.

## Adding New Constrained Properties

To add validation for a new property:

1. **Edit `valid_property_values.json`**:

```json
{
  "node_properties": {
    "gene": {
      "chr": {
        "description": "Chromosome location",
        "values": ["chr1", "chr2", ..., "chr22", "chrX", "chrY", "chrM"],
        "note": "Use 'chr' prefix. Lowercase 'chr', uppercase for X/Y/M."
      }
    }
  },
  "relationship_properties": {
    "part_of_QTL_signal": {
      "tissue": {
        "description": "Tissue type for QTL",
        "values": ["Pancreas", "Islet", "Beta Cell"],
        "note": "Exact tissue names from study data."
      }
    }
  }
}
```

2. **No code changes needed** - the validator automatically picks up new constraints!

## Testing

### Run Validation Tests
```bash
cd PankBaseAgent/text_to_cypher
python3 test_value_validation.py
```

This tests:
- Valid cell type names
- Invalid cell type names (wrong case, hallucinated values)
- Valid disease names
- Invalid disease names (T1D, wrong case)
- Valid regulation values
- Invalid regulation values
- Multiple invalid values in one query

### Run Refinement Tests
```bash
python3 test_refinement.py
```

This now includes tests for invalid property values.

## Example Refinement Flow

**User Query**: "Find upregulated genes in beta cells"

**Iteration 1** (Score: 60/100):
```cypher
MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type)
WHERE ct.name='beta cell' AND deg.UpOrDownRegulation='upregulated'
WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges
RETURN nodes, edges;
```

**Errors**:
- Invalid value 'beta cell' for cell_type.name (-20 points)
- Invalid value 'upregulated' for DEG_in.UpOrDownRegulation (-20 points)

**Refinement Prompt Includes**:
```
Validation feedback:
  1. Invalid value 'beta cell' for cell_type.name. Valid values: ['Acinar Cell', 'Alpha Cell', 'Beta Cell', ...]
  2. Invalid value 'upregulated' for DEG_in.UpOrDownRegulation. Valid values: ['up', 'down']

Detailed Properties for Query Entities:

  cell_type:
    - name (String)
      Valid values: ['Acinar Cell', 'Alpha Cell', 'Beta Cell', 'Delta Cell', 'Ductal Cell', 'Endothelial Cell', 'Macrophage Cell', 'Stellate Cell']
      Note: Case-sensitive. Use exact spelling as shown.

  DEG_in (gene→cell_type):
    - UpOrDownRegulation (String)
      Valid values: ['up', 'down']
      Note: Lowercase only. 'up' means upregulated, 'down' means downregulated.
```

**Iteration 2** (Score: 100/100):
```cypher
MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type)
WHERE ct.name='Beta Cell' AND deg.UpOrDownRegulation='up'
WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges
RETURN nodes, edges;
```

✅ **Both values fixed!**

## Technical Details

### Pattern Matching
The validator uses regex to extract property value comparisons:
```python
# Matches: var.property = 'value' or var.property='value'
value_patterns = re.findall(r'(\w+)\.(\w+)\s*=\s*["\']([^"\']+)["\']', cypher, re.IGNORECASE)
```

### Entity Matching
The validator determines which entity type a variable belongs to by:
1. Extracting all node labels and relationship types from the query
2. Checking if the property exists in the valid values for each entity type
3. Matching the property name to find the correct constraint

### Error Priority
Property value errors are treated as **CRITICAL** because:
- They cause queries to return empty results
- They're easy to fix with the right information
- They're common with small models that hallucinate values

## Files Modified

1. **`src/cypher_validator.py`** (+70 lines)
   - Added `check_property_value_validity()` function
   - Integrated into `validate_cypher()`
   - Updated scoring logic

2. **`test_refinement.py`** (+12 lines)
   - Added test cases for invalid cell type and regulation values

3. **`test_value_validation.py`** (NEW, 200 lines)
   - Comprehensive test suite for property value validation

4. **`VALUE_VALIDATION_SUMMARY.md`** (NEW, this file)
   - Complete documentation of the feature

## Integration with Existing Features

This feature works seamlessly with:

1. **Dynamic Schema Enrichment**: Valid values are shown in the detailed properties during refinement
2. **Iterative Refinement**: Invalid values are caught and fixed in subsequent iterations
3. **Validation Scoring**: Invalid values reduce the score, triggering refinement
4. **Refinement Logging**: Value validation errors are logged in `refinement_metrics.jsonl`

## Future Enhancements

Potential improvements:
1. **Fuzzy Matching**: Suggest closest valid value (e.g., "beta cell" → "Beta Cell")
2. **Pattern Validation**: Validate formats like IDs (e.g., gene IDs must match `ENSG\d+`)
3. **Range Validation**: Check numeric values are in valid ranges (e.g., `pip` between 0 and 1)
4. **Cross-Property Validation**: Check combinations of values (e.g., tissue + cell type)

## Summary

✅ **Property value validation is now fully integrated!**

The validator now:
- Checks property values against `valid_property_values.json`
- Provides detailed error messages with valid values
- Deducts 20 points for invalid values
- Helps the LLM fix errors during refinement
- Works seamlessly with dynamic schema enrichment

This prevents the most common cause of empty query results: using invalid property values.

