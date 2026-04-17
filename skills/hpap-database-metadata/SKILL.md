---
name: hpap-database-query
description: >-
  Query the HPAP Type 1 Diabetes MySQL database using natural language.
  Translates questions into SQL and runs them via pymysql. Use when the user
  asks about HPAP donors, metadata, autoantibodies, cell counts, modalities,
  assay data, or any question about the HPAP T1D research database.
---

# HPAP Database Query

Translate natural language questions into MySQL queries against the HPAP (Human Pancreas Analysis Program) Type 1 Diabetes metadata database.

## Database overview

Two MySQL databases on AWS:

- **`donors`** — 4 tables: donor-level metadata, autoantibody/C-peptide data, cell counts, diagnosis summaries
- **`modalities`** — 20 tables: one per assay type (scRNA-seq, Flow Cytometry, Histology, etc.)

For full table and column definitions, read [schema.md](schema.md).

## Quick start

Connection helper lives in `mai-t1d-metadata-standization/database.py`:

```python
import pymysql
import sys
sys.path.insert(0, "mai-t1d-metadata-standization")
from database import connect

conn = connect("donors")  # or "modalities"
cursor = conn.cursor(pymysql.cursors.DictCursor)
cursor.execute("SELECT `donor_ID`, `clinical_diagnosis` FROM `Metadata` LIMIT 5")
rows = cursor.fetchall()
cursor.close()
conn.close()
```

## Workflow

1. **Read schema.md** to see all tables and columns before writing any query
2. **Identify the right database**: donor-level questions use `donors`, assay-level questions use `modalities`
3. **Translate** the natural language question into a MySQL query
4. **Run** the query using `connect()` + `DictCursor`
5. **Present** results clearly — use tables for tabular data, summaries for aggregations

## Query rules

### Always backtick identifiers

Column and table names contain spaces, parentheses, slashes, and special characters. Always wrap them in backticks:

```sql
SELECT `Donor ID`, `GADA(+/-)`, `C-Peptide (ng/mL)`
FROM `AAb_cPeptide_Metadata`
```

### Donor ID column is inconsistent across tables

| Table | Column name |
|-------|-------------|
| `donors.Metadata` | `donor_ID` and `Donor` (both exist) |
| `donors.AAb_cPeptide_Metadata` | `Donor ID` |
| `donors.cell_counts` | `Donor ID` |
| All `modalities` tables | `Donor` |

When joining across tables, use the correct column name for each side.

### Cross-database queries

Prefix with the database name:

```sql
SELECT m.`donor_ID`, c.`Beta Cell Count`
FROM donors.`Metadata` m
JOIN donors.`cell_counts` c ON m.`donor_ID` = c.`Donor ID`
```

### Modalities tables share a common pattern

Most modality tables have: `Donor`, `Data Modality`, `Tissue`, `File`, `Source`, `Contact`. Some add `Cell_Type`, `Region`, `Run`, `Sample`, or assay-specific fields.

## NL to SQL examples

**"How many donors have T1D?"**
```sql
SELECT COUNT(*) FROM donors.`Metadata`
WHERE `clinical_diagnosis` = 'T1D';
```

**"Show me all donors with GADA positive autoantibodies"**
```sql
SELECT `Donor ID`, `GADA Level`, `GADA(+/-)`
FROM donors.`AAb_cPeptide_Metadata`
WHERE `GADA(+/-)` = '+';
```

**"Which modalities are available for donor HPAP-001?"**
```sql
SELECT `Data Modality`, COUNT(*) as records
FROM modalities.`Overview`
WHERE `Donors` > 0
GROUP BY `Data Modality`;
```

Or check the `Metadata` table's modality flag columns directly:

```sql
SELECT `donor_ID`, `scRNA-seq`, `scATAC-seq`, `Flow Cytometry`,
       `Bulk RNA-seq`, `Histology`, `CODEX`, `IMC`
FROM donors.`Metadata`
WHERE `donor_ID` = 'HPAP-001';
```

**"What is the average beta cell count by diagnosis?"**
```sql
SELECT m.`clinical_diagnosis`, AVG(c.`Beta Cell Count`) as avg_beta
FROM donors.`Metadata` m
JOIN donors.`cell_counts` c ON m.`donor_ID` = c.`Donor ID`
GROUP BY m.`clinical_diagnosis`;
```

## Common pitfalls

1. **Forgetting backticks** — queries will fail on columns like `C-Peptide (ng/mL)` or `GADA(+/-)` without backticks
2. **Wrong donor ID column** — `donor_ID` in Metadata vs `Donor ID` in cell_counts vs `Donor` in modality tables
3. **No LIMIT on exploratory queries** — `Flow_Cytometry` has 1691 rows, `Calcium_Imaging` has 724; always use LIMIT when exploring
4. **Numeric columns stored as VARCHAR** — `age_years`, `BMI`, `C-Peptide (ng/ml)` in `Metadata` are VARCHAR; cast if doing math: `CAST(\`age_years\` AS UNSIGNED)`
5. **Duplicate column names** — `Metadata` has `T1D stage_1` and `T1D stage_2` (disambiguated from duplicate Excel headers)
6. **NULL values** — all columns are nullable; use `IS NOT NULL` or `COALESCE` when needed

For the full list, see [references/hpap-pitfalls.md](references/hpap-pitfalls.md).

## Additional resources

- For real query examples, see [examples/common_queries.sql](examples/common_queries.sql)
- For cross-database join patterns, see [references/cross-database-patterns.md](references/cross-database-patterns.md)
- For the Python helper API, see [scripts/hpap_helper.py](scripts/hpap_helper.py)
