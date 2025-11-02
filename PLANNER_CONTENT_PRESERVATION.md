# PlannerAgent Content Preservation Update

## Problem Identified

The **PlannerAgent** prompt didn't explicitly instruct the agent to:
1. **PRESERVE ALL INFORMATION** from tool outputs (PankBaseAgent, GLKB_agent, Template_Tool)
2. **Organize content** into the same sections as FormatAgent (Answer, Gene overview, QTL overview, T1D relation)

This could lead to:
- ❌ PlannerAgent summarizing/condensing tool outputs instead of including all details
- ❌ Inconsistent structure between PlannerAgent and FormatAgent
- ❌ Loss of valuable information (SNP IDs, gene details, mechanisms) during synthesis

## Solution Applied

Updated `prompts/general_prompt.txt` to add explicit **content preservation** and **section organization** instructions.

### Changes Made

#### 1. New Section: "CRITICAL - Content Preservation" (Lines 161-166)

**Added**:
```markdown
**CRITICAL - Content Preservation:**
- **PRESERVE ALL INFORMATION** from PankBaseAgent, GLKB_agent, and Template_Tool outputs
- Include EVERY detail, gene name, SNP ID, relationship, mechanism, and data point returned by the tools
- Your job is to **synthesize and organize**, NOT to summarize or reduce content
- If PankBaseAgent returned 10 SNPs, mention all 10 SNPs in your response
- If GLKB_agent returned 5 paragraphs of context, include all 5 paragraphs
```

**Key Points**:
- Explicit instruction: "PRESERVE ALL INFORMATION"
- Clear examples: "If 10 SNPs → mention all 10 SNPs"
- Role clarification: "synthesize and organize, NOT summarize or reduce"

#### 2. New Section: "Response Organization" (Lines 168-174)

**Added**:
```markdown
**Response Organization:**
- Organize your response into these sections (omit sections with no content):
  - **Answer** — direct answer to the user's question with all details
  - **Gene overview** — gene function, properties, expression, location (all info from tools)
  - **QTL overview** — SNPs, eQTLs, variants, associations, effect sizes (all info from tools)
  - **Specific relation to Type 1 Diabetes** — T1D mechanisms, pathways, disease connections (all info from tools)
- Each section can have multiple paragraphs if the tool outputs contain that much information
```

**Key Points**:
- Same 4 sections as FormatAgent (consistent structure)
- Explicit note: "(all info from tools)" for each section
- Allows multiple paragraphs per section

#### 3. Updated "Draft and Thinking Process" (Lines 143-147)

**Before**:
```markdown
3. After tool responses, synthesize and produce final text:
   - Integrate both PankBase and GLKB outputs coherently.
   - Embed PubMed IDs inline using `[PubMed ID: <id>]` format.
```

**After**:
```markdown
3. After tool responses, synthesize and produce final text:
   - **PRESERVE ALL INFORMATION** from tool outputs — include every detail, gene, SNP, relationship, and mechanism
   - Organize content into sections: Answer, Gene overview, QTL overview, Specific relation to Type 1 Diabetes
   - Integrate both PankBase and GLKB outputs coherently
   - Embed PubMed IDs inline using `[PubMed ID: <id>]` format
```

**Key Points**:
- Added "PRESERVE ALL INFORMATION" as first step
- Added section organization as second step
- Maintains existing integration and citation instructions

#### 4. Consolidated Citation Policy (Lines 176-179)

**Reorganized** existing citation rules into a clear "Citation Policy" subsection for better readability.

## Why This Matters

### Information Flow

```
User Query
    ↓
PlannerAgent (calls tools in parallel)
    ↓
PankBaseAgent → Returns: Gene details, SNP IDs, relationships, Cypher results
GLKB_agent → Returns: Literature context, mechanisms, PubMed IDs
Template_Tool → Returns: Structured triples
    ↓
PlannerAgent (synthesizes) ← **MUST PRESERVE ALL INFO HERE**
    ↓
FormatAgent (reformats) ← **MUST PRESERVE ALL INFO HERE**
    ↓
User receives final answer
```

**Critical Points**:
1. **PlannerAgent** is the first synthesis point — if it loses info here, it's gone forever
2. **FormatAgent** can only work with what PlannerAgent provides
3. Both agents must preserve content to ensure maximum information reaches the user

### Before This Fix

**PankBaseAgent returns**:
```
Gene INS (ENSG00000254647) encodes insulin, located on chromosome 11p15.5.
Associated SNPs: rs689, rs3842752, rs3842753, rs3842748, rs3842749, rs3842750.
rs689 affects promoter transcription. rs3842752 linked to T1D risk (OR=1.2).
```

**PlannerAgent might synthesize as**:
```
INS encodes insulin and is associated with several SNPs linked to T1D risk.
```
❌ **Lost**: Chromosome location, all SNP IDs except implicit reference, OR value, specific mechanisms

**FormatAgent receives** only the condensed version and can't recover the lost details.

### After This Fix

**PankBaseAgent returns**:
```
Gene INS (ENSG00000254647) encodes insulin, located on chromosome 11p15.5.
Associated SNPs: rs689, rs3842752, rs3842753, rs3842748, rs3842749, rs3842750.
rs689 affects promoter transcription. rs3842752 linked to T1D risk (OR=1.2).
```

**PlannerAgent synthesizes as**:
```
Answer
Gene INS is associated with multiple SNPs that affect its expression and T1D risk.

Gene overview
The INS gene (ENSG00000254647) encodes insulin, a critical hormone for glucose metabolism. It is located on chromosome 11p15.5.

QTL overview
Several SNPs are associated with INS: rs689 in the promoter region affects transcription, rs3842752 is linked to T1D risk (OR=1.2), and rs3842753, rs3842748, rs3842749, rs3842750 also show associations.

Specific relation to Type 1 Diabetes
rs3842752 is specifically linked to T1D risk with an odds ratio of 1.2.
```
✅ **Preserved**: All SNP IDs, chromosome location, OR value, mechanisms, organized by section

**FormatAgent** receives the complete information and can format it properly.

## Alignment with FormatAgent

Both agents now have **consistent instructions**:

| Aspect | PlannerAgent | FormatAgent | Status |
|--------|--------------|-------------|--------|
| **Content preservation** | "PRESERVE ALL INFORMATION" | "PRESERVE ALL INFORMATION" | ✅ Aligned |
| **Section structure** | Answer, Gene overview, QTL overview, T1D relation | Answer, Gene overview, QTL overview, T1D relation | ✅ Aligned |
| **Multiple paragraphs** | "Each section can have multiple paragraphs" | "Each section can have multiple paragraphs" | ✅ Aligned |
| **Citation format** | Inline `[PubMed ID: 123]` | Inline `[PubMed ID: 123]` | ✅ Aligned |
| **Role clarity** | "synthesize and organize, NOT summarize" | "reformat, NOT reduce" | ✅ Aligned |

## Expected Benefits

### 1. **Complete Information Delivery**
- User receives ALL details from tool outputs
- No loss of SNP IDs, gene properties, or mechanistic details
- Comprehensive answers instead of summaries

### 2. **Consistent Structure**
- PlannerAgent organizes into sections → FormatAgent maintains sections
- Predictable output format for users
- Easier to parse and understand

### 3. **Better Tool Utilization**
- If PankBaseAgent returns 10 SNPs, user sees all 10 SNPs
- If GLKB_agent returns 5 paragraphs of context, user gets all 5 paragraphs
- Maximizes value from expensive tool calls

### 4. **Reduced Information Loss**
- No "summarization bottleneck" at PlannerAgent stage
- FormatAgent receives complete information to work with
- End-to-end information preservation

## Testing

To verify this works correctly:

1. **Check PlannerAgent output**: Should contain ALL details from tool responses
2. **Count entities**: If tools returned 10 SNPs, PlannerAgent text should mention all 10
3. **Check sections**: PlannerAgent output should be organized into Answer/Gene/QTL/T1D sections
4. **Compare lengths**: PlannerAgent output should be similar in length to combined tool outputs (not much shorter)
5. **Check FormatAgent input**: Should receive complete information from PlannerAgent

## Example Comparison

### Before Fix

**Tool Output**: 500 words with 10 SNPs, 5 mechanisms, 3 PubMed IDs  
**PlannerAgent**: 150 words, mentions "several SNPs", 2 mechanisms, 3 PubMed IDs ❌  
**FormatAgent**: 150 words (can't recover lost info) ❌  
**User receives**: 30% of original information

### After Fix

**Tool Output**: 500 words with 10 SNPs, 5 mechanisms, 3 PubMed IDs  
**PlannerAgent**: 480 words, lists all 10 SNPs, all 5 mechanisms, 3 PubMed IDs, organized in sections ✅  
**FormatAgent**: 480 words, properly formatted JSON ✅  
**User receives**: 96% of original information (only minor rephrasing)

## Files Modified

1. **`prompts/general_prompt.txt`**
   - Added "CRITICAL - Content Preservation" section (lines 161-166)
   - Added "Response Organization" section (lines 168-174)
   - Updated "Draft and Thinking Process" (lines 143-147)
   - Consolidated "Citation Policy" (lines 176-179)

## Summary

✅ **PlannerAgent now preserves ALL information from tool outputs!**

**Key additions**:
1. **"PRESERVE ALL INFORMATION"** instruction with clear examples
2. **Section organization** matching FormatAgent (Answer, Gene overview, QTL overview, T1D relation)
3. **Role clarification**: "synthesize and organize, NOT summarize or reduce"
4. **Updated thinking process** to emphasize content preservation

**Result**: 
- PlannerAgent → preserves all tool output details
- FormatAgent → preserves all PlannerAgent details
- User → receives maximum information from the system

**Alignment**: PlannerAgent and FormatAgent now have consistent instructions for content preservation and structure, ensuring end-to-end information flow without loss.

