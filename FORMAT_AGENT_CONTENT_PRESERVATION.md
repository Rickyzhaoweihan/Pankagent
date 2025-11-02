# FormatAgent Content Preservation Update

## Problem Identified

The `format_prompt.txt` was too restrictive and could potentially **cut off or summarize** valuable information from the PlannerAgent. The user wants **as much information as possible** — the FormatAgent's job is to **reformat, NOT to reduce content**.

## Solution Applied

Updated `prompts/format_prompt.txt` to emphasize **PRESERVE ALL INFORMATION** throughout the prompt.

### Key Changes

#### 1. Core Responsibilities (Lines 10-14)

**Added**:
```
4. **PRESERVE ALL INFORMATION** from the pre-Final Answer — your job is to reformat, NOT to summarize or cut content.
```

#### 2. Summary Formatting (Lines 37-47)

**Before**:
- "Produce a single essay-style string divided into the four possible sections"
- Implied that content should be condensed into neat paragraphs

**After**:
```
- **CRITICAL**: Include ALL information from the pre-Final Answer. Do NOT summarize, condense, or omit any details.
- **If the pre-Final Answer contains multiple paragraphs of detail, KEEP ALL OF THEM** — just organize them under the appropriate headings.
- If data is missing for a section, omit that section entirely (don't write "No data available" unless the upstream agent explicitly said so).
- You may supplement with **established biomedical facts** but **never remove or condense** information from the pre-Final Answer.
```

#### 3. Format Examples (Lines 49-69)

**Updated** to include explicit instructions in each section:

```
Answer
Your direct answer to the original question. Include ALL details provided by the upstream agent.
If the answer has multiple paragraphs or bullet points, preserve them all.

Gene overview (optional, omit if no content)
ABCD encodes a chloride channel critical for epithelial ion and fluid transport across various tissues. [PubMed ID: 12345678]
Include ALL gene-related information from the pre-Final Answer — function, properties, expression patterns, etc.
If the upstream agent provided multiple paragraphs, keep them all here.

QTL overview (optional, omit if no content)
Several eQTLs have been linked to ABCD expression in pancreatic and pulmonary tissues. [PubMed ID: 34567890]
Include ALL QTL/SNP/variant information from the pre-Final Answer.
List all SNPs, effect sizes, tissues, and any other details provided.

Specific relation to Type 1 Diabetes (optional, omit if no content)
Dysregulation of ABCD expression may influence pancreatic inflammation, contributing indirectly to T1D susceptibility. [PubMed ID: 23456789]
Include ALL T1D-specific information from the pre-Final Answer.
Preserve all mechanistic details, pathway information, and disease connections.
```

#### 4. Content Discipline (Lines 89-95)

**Before**:
```
- Supplement only with verified, non-speculative biomedical information.
- Keep the tone factual, concise, and professional.
```

**After**:
```
- **PRESERVE ALL CONTENT** from the pre-Final Answer — your job is formatting, not summarization.
- Include every detail, data point, gene name, SNP ID, relationship, and explanation provided by the upstream agent.
- You may supplement with verified, non-speculative biomedical information, but NEVER remove or condense existing content.
- Keep the tone factual and professional.
```

#### 5. Quality Enforcement Rules (Lines 170-189)

**Added as Rule #1 (MOST IMPORTANT)**:
```
1. **Content Preservation (MOST IMPORTANT)**
   - **DO NOT summarize, condense, or omit ANY information** from the pre-Final Answer.
   - If the upstream agent provided 5 paragraphs, your output should contain all 5 paragraphs (just organized under headings).
   - If the upstream agent listed 10 SNPs, your output should list all 10 SNPs.
   - Your role is to **reformat**, not to **reduce** content.
```

**Updated Rule #2**:
```
2. **Accuracy First**
   - Never guess or invent biological details.
   - Only write "No data available." if the upstream agent explicitly stated this.
```

**Updated Rule #5**:
```
5. **Formatting Discipline**
   - Organize content under appropriate section headings (Answer, Gene overview, QTL overview, Specific relation to Type 1 Diabetes).
   - Maintain one blank line between each section.
   - Each section can have multiple paragraphs if the content requires it.
```

#### 6. Summary of Required Behavior (Lines 207-214)

**Before**:
```
- Output the summary as one essay-style string with the three required headed sections.
```

**After**:
```
- **PRESERVE ALL INFORMATION** from the pre-Final Answer — reformat, do NOT summarize or condense.
- Organize content under appropriate section headings (Answer, Gene overview, QTL overview, Specific relation to Type 1 Diabetes).
- If the upstream agent provided multiple paragraphs or detailed lists, **keep them all** — just organize under headings.
```

## What This Means

### Before the Fix

FormatAgent might have:
- ❌ Condensed 5 paragraphs into 1 summary paragraph
- ❌ Listed only "several SNPs" instead of naming all 10 SNPs provided
- ❌ Omitted detailed mechanistic explanations to keep output "concise"
- ❌ Removed duplicate or overlapping information

### After the Fix

FormatAgent will:
- ✅ Keep ALL 5 paragraphs (just organize them under section headings)
- ✅ List ALL 10 SNPs with all their details
- ✅ Preserve ALL mechanistic explanations, pathway details, and relationships
- ✅ Keep all information even if it seems redundant (the PlannerAgent included it for a reason)

## Key Principle

**FormatAgent's Role**: 
- **Reformat** ✅ (organize under headings, fix JSON structure, order Cypher queries)
- **NOT Summarize** ❌ (don't condense, don't omit, don't reduce)

Think of it like a **copy editor** who fixes formatting and structure, NOT a **summarizer** who reduces content.

## Example

### PlannerAgent Output (pre-Final Answer):
```
The INS gene encodes insulin, a critical hormone for glucose metabolism. 
It is primarily expressed in pancreatic beta cells. 
The gene is located on chromosome 11 and has multiple transcript variants.

Several SNPs are associated with INS expression:
- rs689 in the promoter region affects transcription
- rs3842752 is linked to T1D risk
- rs3842753 modulates insulin secretion

In T1D, INS is a primary autoimmune target. 
Beta cells producing insulin are destroyed by T cells.
This leads to absolute insulin deficiency.
The HLA region interacts with INS variants to modulate risk.
```

### FormatAgent Output (CORRECT - After Fix):
```json
{
  "summary": "Answer\nThe INS gene is associated with multiple SNPs that affect expression and T1D risk.\n\nGene overview\nThe INS gene encodes insulin, a critical hormone for glucose metabolism. It is primarily expressed in pancreatic beta cells. The gene is located on chromosome 11 and has multiple transcript variants.\n\nQTL overview\nSeveral SNPs are associated with INS expression: rs689 in the promoter region affects transcription, rs3842752 is linked to T1D risk, and rs3842753 modulates insulin secretion.\n\nSpecific relation to Type 1 Diabetes\nIn T1D, INS is a primary autoimmune target. Beta cells producing insulin are destroyed by T cells. This leads to absolute insulin deficiency. The HLA region interacts with INS variants to modulate risk."
}
```

**All information preserved!** ✅

### FormatAgent Output (WRONG - Before Fix):
```json
{
  "summary": "Answer\nThe INS gene is associated with T1D risk.\n\nGene overview\nINS encodes insulin, expressed in beta cells.\n\nQTL overview\nSeveral SNPs affect INS expression and T1D risk.\n\nSpecific relation to Type 1 Diabetes\nINS is an autoimmune target in T1D, leading to insulin deficiency."
}
```

**Information lost!** ❌ (chromosome location, specific SNP IDs, HLA interaction, etc.)

## Files Modified

1. **`prompts/format_prompt.txt`**
   - Added "PRESERVE ALL INFORMATION" emphasis throughout
   - Updated Core Responsibilities (added point #4)
   - Rewrote Summary Formatting section with explicit preservation instructions
   - Updated format examples with detailed preservation notes
   - Rewrote Content Discipline section
   - Added "Content Preservation" as Rule #1 in Quality Enforcement
   - Updated Summary of Required Behavior

## Testing

To verify the fix is working:

1. **Check output length**: FormatAgent output should be similar in length to PlannerAgent output
2. **Count details**: If PlannerAgent mentioned 10 SNPs, FormatAgent output should list all 10
3. **Compare paragraphs**: If PlannerAgent had 5 paragraphs, FormatAgent should preserve all 5 (organized under headings)
4. **Look for omissions**: No gene names, SNP IDs, relationships, or mechanistic details should be missing

## Summary

✅ **FormatAgent now preserves ALL information!**

The prompt has been updated with **7 explicit reminders** to preserve content:
1. Core Responsibilities: "PRESERVE ALL INFORMATION"
2. Summary Formatting: "CRITICAL: Include ALL information"
3. Format Examples: Detailed preservation instructions for each section
4. Content Discipline: "PRESERVE ALL CONTENT"
5. Quality Enforcement Rule #1: "Content Preservation (MOST IMPORTANT)"
6. Summary of Required Behavior: "PRESERVE ALL INFORMATION"
7. Throughout: "never remove or condense", "keep them all", "include every detail"

**Key Message**: FormatAgent is a **formatter**, not a **summarizer**. Its job is to organize content under headings and fix JSON structure, NOT to reduce or condense information.

