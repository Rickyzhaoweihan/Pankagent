# Cross-Database Query Patterns

Patterns for joining across the `donors` and `modalities` databases.

## Basics

Both databases live on the same MySQL server, so cross-database joins work with `database.table` syntax:

```sql
SELECT d.`donor_ID`, m.`Tissue`, m.`Cell_Type`
FROM donors.`Metadata` d
JOIN modalities.`scRNA-seq` m ON d.`donor_ID` = m.`Donor`;
```

## Join key mapping

| donors table | Join column | modalities table | Join column |
|---|---|---|---|
| `Metadata` | `donor_ID` | Any modality table | `Donor` |
| `Metadata` | `Donor` | Any modality table | `Donor` |
| `AAb_cPeptide_Metadata` | `Donor ID` | Any modality table | `Donor` |
| `cell_counts` | `Donor ID` | Any modality table | `Donor` |

The `Metadata` table has both `donor_ID` and `Donor` — they contain the same values. Use whichever matches the other table's column.

## Common patterns

### Donor metadata + modality records

```sql
SELECT d.`donor_ID`, d.`clinical_diagnosis`, d.`sex`,
       r.`Tissue`, r.`Cell_Type`, r.`Source`
FROM donors.`Metadata` d
JOIN modalities.`scRNA-seq` r ON d.`donor_ID` = r.`Donor`
WHERE d.`clinical_diagnosis` = 'T1D';
```

### Autoantibody data + modality records

```sql
SELECT a.`Donor ID`, a.`GADA(+/-)`, a.`C-Peptide (ng/mL)`,
       f.`Cell_Type`, f.`Tissue`
FROM donors.`AAb_cPeptide_Metadata` a
JOIN modalities.`Flow_Cytometry` f ON a.`Donor ID` = f.`Donor`
WHERE a.`GADA(+/-)` = '+';
```

### Cell counts + modality records

```sql
SELECT c.`Donor ID`, c.`Beta Cell Count`, c.`Total Immune Cells`,
       h.`Tissue`, h.`svs_total`
FROM donors.`cell_counts` c
JOIN modalities.`Histology` h ON c.`Donor ID` = h.`Donor`
WHERE c.`Beta Cell Count` IS NOT NULL;
```

### Multi-modality: donors with both scRNA-seq and Flow Cytometry

```sql
SELECT DISTINCT s.`Donor`
FROM modalities.`scRNA-seq` s
WHERE s.`Donor` IN (
    SELECT DISTINCT `Donor` FROM modalities.`Flow_Cytometry`
);
```

### Count modality records per donor across multiple assays

```sql
SELECT d.`donor_ID`,
       (SELECT COUNT(*) FROM modalities.`scRNA-seq` WHERE `Donor` = d.`donor_ID`) AS scrna_records,
       (SELECT COUNT(*) FROM modalities.`Flow_Cytometry` WHERE `Donor` = d.`donor_ID`) AS flow_records,
       (SELECT COUNT(*) FROM modalities.`Histology` WHERE `Donor` = d.`donor_ID`) AS histology_records
FROM donors.`Metadata` d
ORDER BY d.`donor_ID`;
```

### Using the Metadata modality flags instead of joining

The `donors.Metadata` table has DOUBLE columns for each modality (non-null means data exists). This is faster than joining to modality tables:

```sql
SELECT `donor_ID`, `clinical_diagnosis`,
       `scRNA-seq`, `Flow Cytometry`, `Histology`, `CODEX`
FROM donors.`Metadata`
WHERE `scRNA-seq` IS NOT NULL
  AND `Flow Cytometry` IS NOT NULL;
```
