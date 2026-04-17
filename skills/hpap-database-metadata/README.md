# HPAP Database Query Skill

Query the HPAP (Human Pancreas Analysis Program) Type 1 Diabetes MySQL database using natural language. Translates questions into SQL and runs them via pymysql.

## Overview

This skill provides everything needed to query the HPAP T1D metadata databases hosted on AWS. It covers donor-level metadata (clinical diagnosis, autoantibodies, cell counts) and modality-level assay data (scRNA-seq, Flow Cytometry, Histology, CODEX, and 16 more).

Two MySQL databases:

- **`donors`** — 4 tables: donor metadata, autoantibody/C-peptide data, cell counts, diagnosis summaries
- **`modalities`** — 20 tables: one per assay type

## Installation

```bash
pip install pymysql
```

No other dependencies required. The connection helper is included in `scripts/hpap_helper.py`.

## What's Included

### SKILL.md
Comprehensive guide covering the workflow for translating natural language questions into MySQL queries, HPAP-specific query rules (backtick requirements, donor ID inconsistencies across tables), NL-to-SQL examples, and common pitfalls.

### schema.md
Full verbatim `CREATE TABLE` definitions for all 24 tables across both databases, sourced from the original `01_schema.sql`.

### scripts/
- `hpap_helper.py` — Database helper with connection management, query execution, schema inspection, and table preview utilities

### examples/
- `common_queries.sql` — Real HPAP query examples: donor lookups, autoantibody filtering, cell count aggregations, cross-database joins, modality searches

### references/
- `hpap-pitfalls.md` — HPAP-specific gotchas: column naming inconsistencies, VARCHAR numerics, NULL handling, backtick requirements
- `cross-database-patterns.md` — Patterns for joining across `donors` and `modalities` databases

## Quick Start

```python
from scripts.hpap_helper import HPAPDatabase

db = HPAPDatabase()

# List all tables in both databases
db.show_tables()

# Run a query
results = db.query("donors", "SELECT `donor_ID`, `clinical_diagnosis` FROM `Metadata` LIMIT 5")
for row in results:
    print(row)

# Preview a table
db.preview("modalities", "scRNA-seq", n=3)

# Inspect schema
db.describe("donors", "Metadata")
```

### Direct SQL

```sql
-- How many donors have T1D?
SELECT COUNT(*) FROM donors.`Metadata`
WHERE `clinical_diagnosis` = 'T1D';

-- GADA-positive donors
SELECT `Donor ID`, `GADA Level`, `GADA(+/-)`
FROM donors.`AAb_cPeptide_Metadata`
WHERE `GADA(+/-)` = '+';

-- Average beta cell count by diagnosis
SELECT m.`clinical_diagnosis`, AVG(c.`Beta Cell Count`) as avg_beta
FROM donors.`Metadata` m
JOIN donors.`cell_counts` c ON m.`donor_ID` = c.`Donor ID`
GROUP BY m.`clinical_diagnosis`;
```

See `examples/common_queries.sql` for more.

## Requirements

- Python 3.7+
- pymysql
- Network access to the AWS MySQL endpoint
