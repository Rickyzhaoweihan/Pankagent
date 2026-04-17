-- ============================================================================
-- HPAP Common Queries
-- ============================================================================
-- Real query examples for the HPAP T1D metadata databases.
-- All column/table names use backticks due to spaces and special characters.
-- ============================================================================


-- ────────────────────────────────────────────────────────────────────────────
-- DONOR LOOKUPS
-- ────────────────────────────────────────────────────────────────────────────

-- List all donors with their diagnosis
SELECT `donor_ID`, `clinical_diagnosis`, `sex`, `age_years`, `BMI`
FROM donors.`Metadata`
ORDER BY `donor_ID`;

-- Count donors by diagnosis
SELECT `clinical_diagnosis`, COUNT(*) AS n
FROM donors.`Metadata`
GROUP BY `clinical_diagnosis`
ORDER BY n DESC;

-- Find all T1D donors
SELECT `donor_ID`, `sex`, `age_years`, `disease_duration`, `HbA1C (percentage)`
FROM donors.`Metadata`
WHERE `clinical_diagnosis` = 'T1D';

-- Donors with PRS score above a threshold
SELECT `donor_ID`, `clinical_diagnosis`, `prs_score`, `score_sans_hla`
FROM donors.`Metadata`
WHERE `prs_score` > 0.5
ORDER BY `prs_score` DESC;


-- ────────────────────────────────────────────────────────────────────────────
-- AUTOANTIBODY QUERIES
-- ────────────────────────────────────────────────────────────────────────────

-- GADA-positive donors
SELECT `Donor ID`, `Clinical Diagnosis`, `GADA Level`, `GADA(+/-)`
FROM donors.`AAb_cPeptide_Metadata`
WHERE `GADA(+/-)` = '+';

-- Donors positive for 2 or more autoantibodies
SELECT `Donor ID`, `Clinical Diagnosis`, `# AAb Positive`,
       `GADA(+/-)`, `IA-2 (+/-)`, `IAA (+/-)`, `ZnT8 (+/-)`
FROM donors.`AAb_cPeptide_Metadata`
WHERE `# AAb Positive` >= 2;

-- C-Peptide levels by diagnosis
SELECT `Clinical Diagnosis`,
       AVG(`C-Peptide (ng/mL)`) AS avg_cpeptide,
       MIN(`C-Peptide (ng/mL)`) AS min_cpeptide,
       MAX(`C-Peptide (ng/mL)`) AS max_cpeptide
FROM donors.`AAb_cPeptide_Metadata`
GROUP BY `Clinical Diagnosis`;

-- Diagnosis summary (pre-aggregated table)
SELECT * FROM donors.`Summary_by_Diagnosis`;


-- ────────────────────────────────────────────────────────────────────────────
-- CELL COUNT QUERIES
-- ────────────────────────────────────────────────────────────────────────────

-- Average cell counts by diagnosis
SELECT m.`clinical_diagnosis`,
       AVG(c.`Alpha Cell Count`) AS avg_alpha,
       AVG(c.`Beta Cell Count`)  AS avg_beta,
       AVG(c.`Delta Cell Count`) AS avg_delta,
       AVG(c.`Total Immune Cells`) AS avg_immune
FROM donors.`Metadata` m
JOIN donors.`cell_counts` c ON m.`donor_ID` = c.`Donor ID`
GROUP BY m.`clinical_diagnosis`;

-- Donors with highest beta cell counts
SELECT `Donor ID`, `Beta Cell Count`, `Alpha Cell Count`, `GSIR SI`
FROM donors.`cell_counts`
WHERE `Beta Cell Count` IS NOT NULL
ORDER BY `Beta Cell Count` DESC
LIMIT 10;

-- Beta-to-alpha ratio
SELECT `Donor ID`,
       `Beta Cell Count`,
       `Alpha Cell Count`,
       `Beta Cell Count` / NULLIF(`Alpha Cell Count`, 0) AS beta_alpha_ratio
FROM donors.`cell_counts`
WHERE `Beta Cell Count` IS NOT NULL
  AND `Alpha Cell Count` IS NOT NULL
ORDER BY beta_alpha_ratio DESC;


-- ────────────────────────────────────────────────────────────────────────────
-- MODALITY QUERIES
-- ────────────────────────────────────────────────────────────────────────────

-- Overview of all modalities (donor counts and record counts)
SELECT `Data Modality`, `Category`, `Donors`, `Records`, `Contact`
FROM modalities.`Overview`
ORDER BY `Donors` DESC;

-- Which donors have scRNA-seq data?
SELECT DISTINCT `Donor`, `Tissue`, `Cell_Type`
FROM modalities.`scRNA-seq`
ORDER BY `Donor`;

-- Flow cytometry records per donor
SELECT `Donor`, COUNT(*) AS n_records
FROM modalities.`Flow_Cytometry`
GROUP BY `Donor`
ORDER BY n_records DESC;

-- All histology slide counts for a specific donor
SELECT `Donor`, `Tissue`, `svs_total`, `ndpi_total`, `locations`
FROM modalities.`Histology`
WHERE `Donor` = 'HPAP-001';

-- Data tracking: pipeline and storage info per modality
SELECT `Data Modality`, `Pipeline`, `Storage Location (primary)`,
       `Responsible Person`, `Data Status`
FROM modalities.`Data_Track`;


-- ────────────────────────────────────────────────────────────────────────────
-- CROSS-DATABASE JOINS
-- ────────────────────────────────────────────────────────────────────────────

-- Donor diagnosis + their scRNA-seq records
SELECT m.`donor_ID`, m.`clinical_diagnosis`, m.`sex`,
       s.`Tissue`, s.`Cell_Type`
FROM donors.`Metadata` m
JOIN modalities.`scRNA-seq` s ON m.`donor_ID` = s.`Donor`
ORDER BY m.`donor_ID`;

-- T1D donors with Flow Cytometry data
SELECT DISTINCT m.`donor_ID`, m.`age_years`, m.`sex`
FROM donors.`Metadata` m
JOIN modalities.`Flow_Cytometry` f ON m.`donor_ID` = f.`Donor`
WHERE m.`clinical_diagnosis` = 'T1D';

-- Donor cell counts + their available modalities from Metadata flags
SELECT c.`Donor ID`,
       c.`Beta Cell Count`,
       c.`Total Immune Cells`,
       m.`scRNA-seq`, m.`Flow Cytometry`, m.`Histology`, m.`CODEX`
FROM donors.`cell_counts` c
JOIN donors.`Metadata` m ON c.`Donor ID` = m.`donor_ID`
WHERE c.`Beta Cell Count` IS NOT NULL
ORDER BY c.`Beta Cell Count` DESC
LIMIT 20;
