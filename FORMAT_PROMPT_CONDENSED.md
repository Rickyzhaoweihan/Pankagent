# FormatAgent Prompt Condensed for Better Performance

## Problem Identified

The `format_prompt.txt` was **too long and verbose** (214 lines), which can hurt LLM performance:
- Excessive repetition of the same instructions
- Too many examples and explanations
- Redundant quality enforcement rules
- Long-winded descriptions

**Result**: The agent may not follow the format correctly because the prompt is too long to process efficiently.

## Solution Applied

**Condensed the prompt from 214 lines → 114 lines (47% reduction)** while preserving the core instruction: **PRESERVE ALL INFORMATION**.

### What Was Removed

1. **Redundant explanations** - Removed verbose descriptions that repeated the same point
2. **Excessive examples** - Kept one clear example instead of multiple similar ones
3. **Over-detailed quality rules** - Consolidated 7 quality rules into 6 concise output rules
4. **Verbose formatting instructions** - Simplified to essential structure only
5. **Repetitive reminders** - The phrase "PRESERVE ALL INFORMATION" appeared 7+ times, now appears 3 times strategically

### What Was Kept

✅ **Core instruction**: "PRESERVE ALL INFORMATION — do NOT summarize or omit any details"  
✅ **Output structure**: Clear JSON format with template_matching, cypher, and summary fields  
✅ **Section organization**: Answer, Gene overview, QTL overview, Specific relation to Type 1 Diabetes  
✅ **Citation format**: Inline PubMed IDs with space-separation  
✅ **One clear example**: Shows input → output transformation with all info preserved  
✅ **Key principle**: "reformat, NOT reduce"

## Before vs After

### Before (214 lines)
```
### Core Responsibilities

1. Review the pre-Final Answer from upstream agents.  
2. Analyze all Cypher Queries that were generated.  
3. Reformat everything into a clean, factual, and properly structured JSON object that meets all criteria below.  
4. **PRESERVE ALL INFORMATION** from the pre-Final Answer — your job is to reformat, NOT to summarize or cut content.
5. You may supplement the provided data with additional **verified or common biomedical knowledge** only when you are **certain** it is accurate. Treat the input data as authoritative but not exhaustive.

[... 200+ more lines with extensive examples, quality rules, and repetition ...]
```

### After (114 lines)
```
## FormatAgent

You are the **FormatAgent** — the final formatter before output reaches the user.

**Your job**: Reformat the pre-Final Answer into valid JSON. **PRESERVE ALL INFORMATION** — do NOT summarize or omit any details.

[... concise rules, one example, clear structure ...]

### Summary

**Key Principle**: PRESERVE ALL INFORMATION — your job is to reformat, NOT to summarize or reduce content.
```

## Key Changes

### 1. Simplified Header (Lines 1-6)
**Before**: 15 lines of introduction  
**After**: 6 lines with clear job description

### 2. Consolidated Output Rules (Lines 16-60)
**Before**: 7 separate sections with extensive explanations  
**After**: 6 concise rules with bullet points

**Example - Citation Rule**:

**Before** (12 lines):
```
6. **Inline PubMed Citation Enforcement**
   - Only include Pubmed sources, no need to cite any Ensembl sources
   - All PubMed references must appear **inline** within the text as `[PubMed ID: <id>]`. These are individual sources. 
   - If included they must always be at the end of a sentence, never floating individually.  
   - If multiple sources in a sentence, include them as such: `[PubMed ID: <id>] [PubMed ID: <id>] [PubMed ID: <id>]`
   - Never place cirations in lists, arrays, or separate sections.  
   - Bad Examples: `[PubMed ID: <int>; PubMed ID: <int>]` or `[PubMed ID: <int>, PubMed ID: <int>]`
   - Good Examples: `[PubMed ID: <id>] [PubMed ID: <id>]` or `[PubMed ID: <id>]`
```

**After** (5 lines):
```
5. **Citations**
   - Include PubMed IDs inline at end of sentences: `[PubMed ID: 12345678]`
   - Multiple citations: `[PubMed ID: 123] [PubMed ID: 456]` (space-separated, NOT comma/semicolon)
   - Never fabricate IDs
   - No need to cite non-PubMed sources (e.g., Ensembl)
```

### 3. Removed Redundant Quality Rules Section
**Before**: 40+ lines of "Quality Enforcement Rules" that repeated the same instructions  
**After**: Integrated into the 6 output rules above

### 4. Simplified Example Section (Lines 78-106)
**Before**: 2 examples with extensive annotations  
**After**: 1 clear example showing input → output with preserved content

### 5. Concise Summary (Lines 110-114)
**Before**: 20+ lines of "Summary of Required Behavior"  
**After**: 5 bullet points with key principle emphasized

## Performance Benefits

### Token Reduction
- **Before**: ~2,500 tokens (estimated)
- **After**: ~1,200 tokens (estimated)
- **Savings**: ~52% reduction in prompt tokens

### Clarity Improvement
- **Before**: Key instruction buried in verbose text
- **After**: "PRESERVE ALL INFORMATION" appears prominently 3 times (header, rule #1, summary)

### Processing Efficiency
- **Before**: Agent must parse 214 lines to find relevant instructions
- **After**: Agent sees clear structure in 114 lines

## What This Means for the Agent

### Before (Long Prompt)
- ❌ May miss key instructions buried in verbose text
- ❌ Slower processing due to long context
- ❌ May focus on wrong details (e.g., citation formatting) instead of content preservation
- ❌ Higher cost per API call

### After (Concise Prompt)
- ✅ Clear, prominent instruction: "PRESERVE ALL INFORMATION"
- ✅ Faster processing with focused rules
- ✅ Better adherence to format (less confusion)
- ✅ Lower cost per API call

## Testing

To verify the condensed prompt works correctly:

1. **Check content preservation**: Output should still contain ALL details from pre-Final Answer
2. **Check format compliance**: JSON structure should be correct
3. **Check section organization**: Content should be organized under appropriate headings
4. **Check citation format**: PubMed IDs should be inline and space-separated

## Files Modified

1. **`prompts/format_prompt.txt`**
   - Reduced from 214 → 114 lines (47% reduction)
   - Removed redundant explanations and quality rules
   - Kept core instruction: "PRESERVE ALL INFORMATION"
   - Simplified to 6 concise output rules + 1 clear example

2. **`FORMAT_PROMPT_CONDENSED.md`** (NEW)
   - Documentation of changes and rationale

## Summary

✅ **Prompt condensed by 47% (214 → 114 lines)**

**Key improvements**:
- Removed redundant explanations and repetitive quality rules
- Consolidated 7+ sections into 6 concise output rules
- Kept one clear example instead of multiple similar ones
- Maintained core instruction: "PRESERVE ALL INFORMATION" (appears 3 times strategically)

**Expected benefits**:
- Faster processing (52% fewer tokens)
- Better format compliance (clearer instructions)
- Lower API costs
- Same content preservation guarantee

**Key Principle Preserved**: The agent's job is to **reformat, NOT reduce** content. This instruction is now more prominent and easier to follow.

