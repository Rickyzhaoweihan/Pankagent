# HPAP Database Pitfalls

Common gotchas when querying the HPAP T1D metadata databases.

## 1. Backticks are mandatory

Column and table names contain spaces, parentheses, slashes, hyphens, and Unicode characters. Every identifier must be wrapped in backticks.

```sql
-- WRONG: will fail
SELECT Donor ID, GADA(+/-) FROM AAb_cPeptide_Metadata;

-- CORRECT
SELECT `Donor ID`, `GADA(+/-)` FROM `AAb_cPeptide_Metadata`;
```

Notable examples: `C-Peptide (ng/mL)`, `GADA Ratio (level/cutoff)`, `IA-2 (+/-)`, `≥1 AAb+`, `Tissue / Data Modality`, `Storage Location (primary)`, `Other / Notes`.

## 2. Donor ID column naming is inconsistent

The donor identifier column has a different name in almost every table:

| Table | Column |
|-------|--------|
| `donors.Metadata` | `donor_ID` (also has a separate `Donor` column) |
| `donors.AAb_cPeptide_Metadata` | `Donor ID` |
| `donors.cell_counts` | `Donor ID` |
| `donors.Summary_by_Diagnosis` | (no donor column — aggregated by diagnosis) |
| All `modalities.*` tables | `Donor` |

When joining, always check which column name each side uses:

```sql
-- Metadata (donor_ID) joined with cell_counts (Donor ID)
SELECT m.`donor_ID`, c.`Beta Cell Count`
FROM donors.`Metadata` m
JOIN donors.`cell_counts` c ON m.`donor_ID` = c.`Donor ID`;
```

## 3. Numeric columns stored as VARCHAR

Several columns in `donors.Metadata` look numeric but are stored as `VARCHAR` due to mixed values in the original Excel:

- `age_years` — VARCHAR(50)
- `BMI` — VARCHAR(50)
- `C-Peptide (ng/ml)` — VARCHAR(50)
- `HbA1C (percentage)` — VARCHAR(50)
- `disease_duration` — VARCHAR(100)
- `n_autoantibodies` — VARCHAR(50)

If doing math, cast explicitly:

```sql
SELECT `donor_ID`, CAST(`age_years` AS UNSIGNED) AS age
FROM donors.`Metadata`
WHERE `age_years` REGEXP '^[0-9]+$'
ORDER BY age DESC;
```

## 4. Duplicate column names from Excel

Excel allowed duplicate column headers. These were disambiguated with `_1`, `_2` suffixes:

- `Metadata` has `T1D stage_1` and `T1D stage_2` (originally both "T1D stage")
- `Data_Track` has `Unnamed: 10` (artifact from Excel import)

## 5. All columns are nullable

No `NOT NULL` constraints exist. Every column in every table allows NULL. Always guard against NULLs:

```sql
-- Use IS NOT NULL
SELECT `Donor ID`, `Beta Cell Count`
FROM donors.`cell_counts`
WHERE `Beta Cell Count` IS NOT NULL;

-- Use COALESCE for defaults
SELECT `Donor ID`, COALESCE(`GSIR SI`, 0) AS gsir
FROM donors.`cell_counts`;
```

## 6. No primary keys or foreign keys

Tables have no PRIMARY KEY, no FOREIGN KEY constraints, and no indexes. This means:

- Duplicate rows are possible
- Joins are not enforced by the database
- Full table scans on every query (small dataset, so performance is fine)

## 7. Always use LIMIT for exploration

Some tables are large enough to produce unwieldy output:

| Table | Rows |
|-------|------|
| `Flow_Cytometry` | 1,691 |
| `Calcium_Imaging` | 724 |
| `Histology` | 708 |
| `BCR-seq` | 320 |

Always add `LIMIT` when exploring.

## 8. Cross-database queries need explicit prefixes

To query across `donors` and `modalities` in a single statement, prefix table names with the database:

```sql
SELECT m.`donor_ID`, s.`Tissue`
FROM donors.`Metadata` m
JOIN modalities.`scRNA-seq` s ON m.`donor_ID` = s.`Donor`;
```
