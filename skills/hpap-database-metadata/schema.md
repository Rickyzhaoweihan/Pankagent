# HPAP Database Schema Reference

Auto-generated with distinct values for every column.

## Database: `donors`

4 tables.

### `AAb_cPeptide_Metadata` (193 rows)

```
CREATE TABLE `AAb_cPeptide_Metadata` (
  `Donor ID` VARCHAR(50),
  `Clinical Diagnosis` VARCHAR(255),
  `Sex` VARCHAR(50),
  `Age (yr)` BIGINT,
  `Race` VARCHAR(100),
  `C-Peptide (ng/mL)` DOUBLE,
  `GADA Level` DOUBLE,
  `GADA Cutoff` DOUBLE,
  `GADA Ratio (level/cutoff)` DOUBLE,
  `GADA(+/-)` VARCHAR(10),
  `IA-2 Level` DOUBLE,
  `IA-2 Cutoff` DOUBLE,
  `IA-2 Ratio` DOUBLE,
  `IA-2 (+/-)` VARCHAR(10),
  `IAA Level` DOUBLE,
  `IAA Cutoff` DOUBLE,
  `IAA Ratio` DOUBLE,
  `IAA (+/-)` VARCHAR(10),
  `ZnT8 Level` DOUBLE,
  `ZnT8 Cutoff` DOUBLE,
  `ZnT8 Ratio` DOUBLE,
  `ZnT8 (+/-)` VARCHAR(10),
  `# AAb Positive` BIGINT
);
```

**Column values:**

- **`Donor ID`** (31 distinct): `HPAP-001`, `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-025`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-030` ... (>30 distinct values)
- **`Clinical Diagnosis`** (11 distinct): `Recent T1DM Unsuspected`, `T1D control`, `T1DM`, `T1DM (recent DKA)`, `T1DM or MODY, undetermined`, `T1DM Recent onset`, `T2D control`, `T2DM`, `T2DM (? prediabetic)`, `T2DM Gastric bypass`, `T2DM polycystic ovaries`
- **`Sex`** (2 distinct): `Female`, `Male`
- **`Age (yr)`** (31 distinct): `1`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15`, `17`, `18`, `19`, `20`, `21`, `22`, `23`, `24`, `25`, `26`, `27`, `28`, `29`, `30`, `31`, `32` ... (>30 distinct values)
- **`Race`** (6 distinct): `African American`, `American Indian or Alaska Native`, `Asian`, `Biracial`, `Caucasian`, `Hispanic`
- **`C-Peptide (ng/mL)`** (31 distinct): `0.0`, `0.02`, `0.03`, `0.05`, `0.06`, `0.07`, `0.08`, `0.09`, `0.1`, `0.12`, `0.13`, `0.16`, `0.17`, `0.2`, `0.21`, `0.23`, `0.25`, `0.26`, `0.27`, `0.3`, `0.37`, `0.43`, `0.51`, `0.54`, `0.56`, `0.62`, `0.67`, `0.69`, `0.79`, `0.97` ... (>30 distinct values)
- **`GADA Level`** (31 distinct): `0.0`, `1.0`, `2.0`, `3.0`, `5.0`, `6.0`, `8.0`, `9.0`, `12.0`, `13.0`, `15.0`, `18.0`, `23.0`, `24.0`, `27.0`, `29.0`, `33.0`, `37.0`, `38.0`, `55.0`, `79.0`, `84.0`, `95.0`, `96.0`, `114.0`, `119.0`, `161.0`, `165.0`, `202.0`, `203.0` ... (>30 distinct values)
- **`GADA Cutoff`** (2 distinct): `0.0`, `20.0`
- **`GADA Ratio (level/cutoff)`** (31 distinct): `0.0`, `0.05`, `0.1`, `0.15`, `0.25`, `0.3`, `0.4`, `0.45`, `0.6`, `0.65`, `0.75`, `0.9`, `1.15`, `1.2`, `1.35`, `1.45`, `1.65`, `1.85`, `1.9`, `2.75`, `3.95`, `4.2`, `4.75`, `4.8`, `5.7`, `5.95`, `8.05`, `8.25`, `10.1`, `10.15` ... (>30 distinct values)
- **`GADA(+/-)`** (2 distinct): `-`, `+`
- **`IA-2 Level`** (20 distinct): `0.0`, `1.0`, `3.0`, `5.0`, `7.0`, `13.0`, `18.0`, `21.0`, `31.0`, `49.0`, `55.0`, `56.0`, `64.0`, `112.0`, `121.0`, `139.0`, `155.0`, `207.0`, `245.0`, `255.0`
- **`IA-2 Cutoff`** (2 distinct): `0.0`, `5.0`
- **`IA-2 Ratio`** (20 distinct): `0.0`, `0.2`, `0.6`, `1.0`, `1.4`, `2.6`, `3.6`, `4.2`, `6.2`, `9.8`, `11.0`, `11.2`, `12.8`, `22.4`, `24.2`, `27.8`, `31.0`, `41.4`, `49.0`, `51.0`
- **`IA-2 (+/-)`** (2 distinct): `-`, `+`
- **`IAA Level`** (31 distinct): `-0.008`, `-0.007`, `-0.006`, `-0.005`, `-0.004`, `-0.003`, `-0.002`, `-0.001`, `0.0`, `0.001`, `0.002`, `0.003`, `0.004`, `0.005`, `0.006`, `0.007`, `0.013`, `0.014`, `0.0161`, `0.02`, `0.024`, `0.031`, `0.038`, `0.041`, `0.055`, `0.056`, `0.068`, `0.071`, `0.123`, `0.147` ... (>30 distinct values)
- **`IAA Cutoff`** (3 distinct): `0.0`, `0.001`, `0.01`
- **`IAA Ratio`** (31 distinct): `-0.8`, `-0.7`, `-0.6`, `-0.5`, `-0.4`, `-0.3`, `-0.2`, `-0.1`, `0.0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`, `1.3`, `1.4`, `1.61`, `2.0`, `2.4`, `3.1`, `3.8`, `4.1`, `5.5`, `5.6`, `6.8`, `7.1`, `12.3`, `14.7` ... (>30 distinct values)
- **`IAA (+/-)`** (2 distinct): `-`, `+`
- **`ZnT8 Level`** (31 distinct): `-0.015`, `-0.012`, `-0.01`, `-0.008`, `-0.007`, `-0.006`, `-0.005`, `-0.004`, `-0.003`, `-0.002`, `-0.001`, `0.0`, `0.001`, `0.002`, `0.003`, `0.004`, `0.005`, `0.006`, `0.007`, `0.008`, `0.009`, `0.011`, `0.013`, `0.023`, `0.027`, `0.031`, `0.038`, `0.071`, `0.073`, `0.087` ... (>30 distinct values)
- **`ZnT8 Cutoff`** (3 distinct): `0.0`, `0.01`, `0.02`
- **`ZnT8 Ratio`** (31 distinct): `-0.75`, `-0.6`, `-0.5`, `-0.4`, `-0.35`, `-0.3`, `-0.25`, `-0.2`, `-0.15`, `-0.1`, `-0.05`, `0.0`, `0.05`, `0.1`, `0.15`, `0.2`, `0.25`, `0.3`, `0.35`, `0.4`, `0.45`, `0.55`, `0.65`, `1.15`, `1.35`, `1.55`, `1.9`, `3.55`, `3.65`, `4.35` ... (>30 distinct values)
- **`ZnT8 (+/-)`** (2 distinct): `-`, `+`
- **`# AAb Positive`** (5 distinct): `0`, `1`, `2`, `3`, `4`

### `Metadata` (57 rows)

```
CREATE TABLE `Metadata` (
  `donor_ID` VARCHAR(50),
  `clinical_diagnosis` VARCHAR(255),
  `Derived Diabetes Status` VARCHAR(255),
  `gada` VARCHAR(50),
  `ia_2` VARCHAR(50),
  `iaa` VARCHAR(50),
  `znt8` VARCHAR(50),
  `T1D stage_1` VARCHAR(100),
  `n_autoantibodies` VARCHAR(50),
  `disease_duration` VARCHAR(100),
  `age_years` VARCHAR(50),
  `HbA1C (percentage)` VARCHAR(50),
  `prs_score` DOUBLE,
  `score_sans_hla` DOUBLE,
  `sex` VARCHAR(50),
  `Donor` VARCHAR(50),
  `DiseaseStatus` VARCHAR(100),
  `BMI` VARCHAR(50),
  `C-Peptide (ng/ml)` VARCHAR(50),
  `Diabetes Status` VARCHAR(100),
  `Family History of Diabetes` VARCHAR(255),
  `Genetic Ancestry (PancDB)` VARCHAR(255),
  `Ethnicities` VARCHAR(255),
  `T1D stage_2` VARCHAR(100),
  `scRNA-seq` DOUBLE,
  `scATAC-seq` DOUBLE,
  `snMultiomics` DOUBLE,
  `CITE-seq Protein` DOUBLE,
  `TEA-seq` DOUBLE,
  `BCR-seq` DOUBLE,
  `TCR-seq` DOUBLE,
  `Bulk RNA-seq` DOUBLE,
  `Bulk ATAC-seq` DOUBLE,
  `WGS` DOUBLE,
  `Calcium Imaging` DOUBLE,
  `Flow Cytometry` DOUBLE,
  `Oxygen Consumption` DOUBLE,
  `Perifusion` DOUBLE,
  `CODEX` BIGINT,
  `IMC` DOUBLE,
  `Histology` BIGINT,
  `Notes` TEXT
);
```

**Column values:**

- **`donor_ID`** (31 distinct): `HPAP-001`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-009`, `HPAP-010`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-025`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-030`, `HPAP-039`, `HPAP-040`, `HPAP-041`, `HPAP-042`, `HPAP-043`, `HPAP-044`, `HPAP-045`, `HPAP-046`, `HPAP-047` ... (>30 distinct values)
- **`clinical_diagnosis`** (7 distinct): `???`, `ND`, `T1DM`, `T1DM Recent`, `T1DM/MODY`, `T2DM`, `T2DM Gastric Bypass`
- **`Derived Diabetes Status`** (4 distinct): `Diabetes`, `Normal`, `Prediabetes`, `Unknown`
- **`gada`** (3 distinct): `0`, `1`, `Unknown`
- **`ia_2`** (3 distinct): `0`, `1`, `Unknown`
- **`iaa`** (3 distinct): `0`, `1`, `Unknown`
- **`znt8`** (3 distinct): `0`, `1`, `Unknown`
- **`T1D stage_1`** (5 distinct): `No stage`, `not applicable`, `Stage 2: Two or more autoantibodies, dysglycemia (e.g., HbA1c â‰¥ 5.7%)`, `Stage 3: One or more autoantibodies and diagnostic hyperglycemia or T1D diagn...`, `Unknown`
- **`n_autoantibodies`** (9 distinct): `Non-0`, `Non-1`, `Non-2`, `T1D-0`, `T1D-1`, `T1D-2`, `T1D-4`, `T2D`, `Unknown`
- **`disease_duration`** (17 distinct): `<5 years`, `12 years`, `14 years`, `18 years`, `2 years`, `2-3 years`, `2.5 years`, `20 years`, `22 years`, `3 years`, `4 years`, `5 years`, `5-6 years`, `6 years`, `7 years`, `8 years`, `Unknown`
- **`age_years`** (31 distinct): `1.08`, `11`, `13`, `14`, `15`, `17`, `18`, `19`, `21`, `22`, `23`, `24`, `27`, `29`, `3`, `30`, `31`, `35`, `38`, `39`, `4`, `40`, `41`, `42`, `43`, `46`, `47`, `48`, `5`, `50` ... (>30 distinct values)
- **`HbA1C (percentage)`** (31 distinct): `10.2`, `10.8`, `12.4`, `4.4`, `4.7`, `4.8`, `4.9`, `5`, `5.1`, `5.2`, `5.3`, `5.4`, `5.5`, `5.6`, `5.7`, `5.8`, `5.9`, `6`, `6.3`, `6.7`, `6.8`, `7.1`, `7.3`, `7.5`, `7.6`, `7.8`, `8.1`, `8.2`, `8.3`, `8.9` ... (>30 distinct values)
- **`prs_score`** (31 distinct): `-0.0106504`, `-0.0089708`, `-0.00731533`, `-0.00723869`, `-0.00656241`, `-0.00651702`, `-0.00604343`, `-0.00477226`, `-0.0047635`, `-0.00459839`, `-0.00431423`, `-0.00430219`, `-0.00396861`, `-0.00396606`, `-0.00293832`, `-0.00274051`, `-0.00253796`, `-0.00199672`, `-0.00191919`, `-0.000972628`, `-0.000586861`, `-0.000426277`, `-0.000241971`, `-6.5301e-05`, `0.0`, `1.8036e-05`, `0.000448905`, `0.00118978`, `0.00128431`, `0.00245474` ... (>30 distinct values)
- **`score_sans_hla`** (31 distinct): `-0.010828`, `-0.010819`, `-0.0106813`, `-0.0106502`, `-0.0101601`, `-0.0100063`, `-0.00917164`, `-0.00881007`, `-0.0079903`, `-0.00791104`, `-0.00767649`, `-0.0076056`, `-0.00758806`, `-0.0074791`, `-0.00694328`, `-0.00670933`, `-0.00666292`, `-0.00641231`, `-0.0063847`, `-0.00625784`, `-0.00593044`, `-0.00582015`, `-0.0056037`, `-0.00548993`, `-0.00463582`, `-0.00439851`, `-0.00434216`, `-0.0042097`, `-0.00405485`, `-0.0040041` ... (>30 distinct values)
- **`sex`** (3 distinct): `Female`, `Male`, `Unknown`
- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-009`, `HPAP-010`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-025`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-030`, `HPAP-039`, `HPAP-040`, `HPAP-041`, `HPAP-042`, `HPAP-043`, `HPAP-044`, `HPAP-045`, `HPAP-046`, `HPAP-047` ... (>30 distinct values)
- **`DiseaseStatus`** (20 distinct): `No HX DIAB`, `No HX DIAB (Asthma)`, `No HX Diabetes`, `Recent T1DM (Unsuspected)`, `T1DM`, `T1DM (12 yrs duration)`, `T1DM (14 yrs duration)`, `T1DM (3 yrs duration)`, `T1DM (5 yrs duration)`, `T1DM (6 yrs duration)`, `T1DM (7 yrs duration)`, `T1DM (8 yr duration)`, `T1DM or MODY (20 yrs, undetermined)`, `T2DM`, `T2DM (<5yrs)`, `T2DM (18 yrs Gastric Bypass)`, `T2DM (2 yrs duration)`, `T2DM (4 yrs Gastric Bypass)`, `T2DM (6 yrs duration)`, `Unknown`
- **`BMI`** (31 distinct): `12`, `13.2`, `14.9`, `16.3`, `16.82`, `17.3`, `17.9`, `19.1`, `20`, `20.8`, `20.93`, `20.96`, `21.3`, `21.35`, `21.4`, `23.7`, `23.98`, `24.07`, `24.1`, `24.3`, `24.47`, `25.03`, `25.92`, `26.2`, `26.65`, `27.52`, `28.6`, `28.72`, `28.99`, `29.45` ... (>30 distinct values)
- **`C-Peptide (ng/ml)`** (31 distinct): `0`, `0.02`, `0.08`, `0.09`, `0.17`, `0.23`, `0.25`, `0.3`, `0.37`, `0.43`, `0.56`, `0.69`, `0.79`, `0.97`, `1.14`, `1.24`, `1.44`, `1.7`, `1.88`, `1.92`, `11.97`, `12.5`, `13.29`, `18.4`, `2.24`, `2.59`, `2.72`, `2.76`, `20.74`, `3.19` ... (>30 distinct values)
- **`Diabetes Status`** (3 distinct): ``, `/phenotype-terms/MONDO_0005147/`, `/phenotype-terms/MONDO_0005148/`
- **`Family History of Diabetes`** (4 distinct): ``, `-`, `False`, `True`
- **`Genetic Ancestry (PancDB)`** (5 distinct): ``, `African american/Black`, `Asian`, `Caucasian`, `Hispanic`
- **`Ethnicities`** (5 distinct): ``, `African American`, `Asian`, `Hispanic`, `White`
- **`T1D stage_2`** (6 distinct): ``, `At-risk: single or transient autoantibody, normal glucose level`, `Conflicting diabetes evidence`, `No sufficient information to derive`, `Stage 2: two or more autoantibodies, dysglycemia (e.g., HbA1c 芒鈥?5.7%)`, `Stage 3: one or more autoantibodies and diagnostic hyperglycemia or T1D diagn...`
- **`scRNA-seq`** (2 distinct): `0.0`, `1.0`
- **`scATAC-seq`** (2 distinct): `0.0`, `1.0`
- **`snMultiomics`** (2 distinct): `0.0`, `1.0`
- **`CITE-seq Protein`** (2 distinct): `0.0`, `1.0`
- **`TEA-seq`** (2 distinct): `0.0`, `1.0`
- **`BCR-seq`** (2 distinct): `0.0`, `1.0`
- **`TCR-seq`** (2 distinct): `0.0`, `1.0`
- **`Bulk RNA-seq`** (2 distinct): `0.0`, `1.0`
- **`Bulk ATAC-seq`** (2 distinct): `0.0`, `1.0`
- **`WGS`** (2 distinct): `0.0`, `1.0`
- **`Calcium Imaging`** (2 distinct): `0.0`, `1.0`
- **`Flow Cytometry`** (2 distinct): `0.0`, `1.0`
- **`Oxygen Consumption`** (2 distinct): `0.0`, `1.0`
- **`Perifusion`** (2 distinct): `0.0`, `1.0`
- **`CODEX`** (2 distinct): `0`, `1`
- **`IMC`** (2 distinct): `0.0`, `1.0`
- **`Histology`** (2 distinct): `0`, `1`
- **`Notes`** (27 distinct): ``, `"Cohort_ID:HPAP-187;hypertension, hyperlipidemia, moyamoya, cellulitis, aller...`, `Cohort_ID:HPAP-025`, `Cohort_ID:HPAP-030;addison's disease, hypothyroidism, autoimmune polyglandula...`, `Cohort_ID:HPAP-041;celiac disease`, `Cohort_ID:HPAP-046;asthma`, `Cohort_ID:HPAP-048;substance abuse disorder`, `Cohort_ID:HPAP-073`, `Cohort_ID:HPAP-076;congestive heart failure, cardiomyopathy"
HPAP-077,ND,Pre...`, `Cohort_ID:HPAP-094;skin cancer, deep vein thrombosis, transient ischemic atta...`, `Cohort_ID:HPAP-115;multiple sclerosis`, `Cohort_ID:HPAP-116;obesity`, `Cohort_ID:HPAP-149;-`, `Cohort_ID:HPAP-150;atrial fibrillation, hyperlipidemia, hyperthyroidism, card...`, `Cohort_ID:HPAP-171;-`, `Cohort_ID:HPAP-172;-`, `Cohort_ID:HPAP-173;-`, `Cohort_ID:HPAP-174;kidney stones , edema, prolapsed uterus"
HPAP-175,ND,Norm...`, `other disease states: Autoimmune disease, Hyperlipidemia, Hypertension"
HPAP...`, `other disease states: CAD, Hyperlipidemia, Hypertension"
HPAP-002,T1DM,Diabe...`, `other disease states: CAD, Hyperlipidemia, Hypertension"
HPAP-008,ND,Normal,...`, `other disease states: CAD, Hyperlipidemia, Hypertension"
HPAP-052,ND,Normal,...`, `other disease states: Hyperlipidemia, Hypertension"
HPAP-011,ND,Normal,0,0,0...`, `other disease states: Hyperlipidemia, Hypertension"
HPAP-062,T2DM,Unknown,0,...`, `other disease states: Hyperlipidemia, Hypertension"
HPAP-121,ND,Normal,0,0,0...`, `other disease states: Hypertension`, `other disease states: Hypothyroidism`

### `Summary_by_Diagnosis` (11 rows)

```
CREATE TABLE `Summary_by_Diagnosis` (
  `Clinical Diagnosis` VARCHAR(255),
  `N` BIGINT,
  `C-Peptide Mean` DOUBLE,
  `C-Peptide Median` DOUBLE,
  `GADA+` BIGINT,
  `IA-2+` BIGINT,
  `IAA+` BIGINT,
  `ZnT8+` BIGINT,
  `≥1 AAb+` BIGINT,
  `≥2 AAb+` BIGINT,
  `≥3 AAb+` BIGINT
);
```

**Column values:**

- **`Clinical Diagnosis`** (11 distinct): `Recent T1DM Unsuspected`, `T1D control`, `T1DM`, `T1DM (recent DKA)`, `T1DM or MODY, undetermined`, `T1DM Recent onset`, `T2D control`, `T2DM`, `T2DM (? prediabetic)`, `T2DM Gastric bypass`, `T2DM polycystic ovaries`
- **`N`** (7 distinct): `1`, `2`, `3`, `38`, `39`, `48`, `58`
- **`C-Peptide Mean`** (11 distinct): `0.37`, `0.43`, `0.5`, `0.69`, `1.13`, `3.28`, `4.66`, `6.92`, `7.69`, `9.43`, `11.16`
- **`C-Peptide Median`** (11 distinct): `0.06`, `0.37`, `0.43`, `0.69`, `1.13`, `2.24`, `4.66`, `6.1`, `6.29`, `9.43`, `11.16`
- **`GADA+`** (5 distinct): `0`, `1`, `4`, `13`, `20`
- **`IA-2+`** (4 distinct): `0`, `1`, `4`, `11`
- **`IAA+`** (4 distinct): `0`, `1`, `2`, `20`
- **`ZnT8+`** (5 distinct): `0`, `1`, `2`, `3`, `4`
- **`≥1 AAb+`** (6 distinct): `0`, `1`, `2`, `6`, `21`, `28`
- **`≥2 AAb+`** (4 distinct): `0`, `1`, `4`, `15`
- **`≥3 AAb+`** (4 distinct): `0`, `1`, `2`, `4`

### `cell_counts` (193 rows)

```
CREATE TABLE `cell_counts` (
  `Donor ID` VARCHAR(50),
  `Alpha Cell Count` BIGINT,
  `Beta Cell Count` BIGINT,
  `Delta Cell Count` BIGINT,
  `Epsilon Cell Count` BIGINT,
  `PP Cell Count` BIGINT,
  `T Cell Count` BIGINT,
  `B Cell Count` BIGINT,
  `Macrophage Count` BIGINT,
  `NK Cell Count` BIGINT,
  `Total Immune Cells` BIGINT,
  `GSIR SI` DOUBLE
);
```

**Column values:**

- **`Donor ID`** (31 distinct): `HPAP-001`, `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-025`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-030` ... (>30 distinct values)
- **`Alpha Cell Count`** (31 distinct): `0`, `65`, `150`, `172`, `252`, `257`, `325`, `404`, `517`, `584`, `605`, `724`, `820`, `844`, `885`, `994`, `1001`, `1048`, `1450`, `1584`, `1617`, `1681`, `1776`, `1963`, `2213`, `2364`, `2425`, `2747`, `2778`, `3046` ... (>30 distinct values)
- **`Beta Cell Count`** (31 distinct): `0`, `28`, `56`, `76`, `105`, `119`, `158`, `177`, `239`, `249`, `272`, `362`, `456`, `496`, `499`, `518`, `598`, `607`, `632`, `653`, `655`, `719`, `831`, `887`, `959`, `1266`, `1331`, `1368`, `1676`, `1698` ... (>30 distinct values)
- **`Delta Cell Count`** (31 distinct): `0`, `21`, `23`, `24`, `32`, `41`, `44`, `51`, `53`, `56`, `58`, `75`, `79`, `80`, `104`, `136`, `161`, `192`, `219`, `267`, `268`, `362`, `366`, `368`, `395`, `400`, `411`, `455`, `462`, `735` ... (>30 distinct values)
- **`Epsilon Cell Count`** (31 distinct): `0`, `1`, `2`, `4`, `5`, `6`, `10`, `11`, `12`, `15`, `16`, `17`, `19`, `20`, `22`, `24`, `26`, `30`, `32`, `33`, `34`, `36`, `37`, `40`, `42`, `51`, `54`, `56`, `57`, `65` ... (>30 distinct values)
- **`PP Cell Count`** (31 distinct): `0`, `1`, `3`, `4`, `7`, `25`, `41`, `45`, `51`, `52`, `54`, `80`, `83`, `99`, `104`, `105`, `108`, `143`, `159`, `188`, `193`, `227`, `246`, `275`, `281`, `327`, `343`, `347`, `355`, `359` ... (>30 distinct values)
- **`T Cell Count`** (31 distinct): `0`, `1`, `2`, `4`, `8`, `9`, `12`, `13`, `19`, `22`, `25`, `26`, `27`, `29`, `31`, `33`, `37`, `38`, `40`, `43`, `47`, `48`, `49`, `53`, `54`, `62`, `65`, `67`, `74`, `79` ... (>30 distinct values)
- **`B Cell Count`** (31 distinct): `0`, `1`, `2`, `3`, `4`, `5`, `6`, `8`, `9`, `10`, `11`, `12`, `13`, `16`, `17`, `18`, `21`, `22`, `28`, `30`, `31`, `32`, `33`, `34`, `35`, `36`, `37`, `41`, `47`, `49` ... (>30 distinct values)
- **`Macrophage Count`** (31 distinct): `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `11`, `15`, `18`, `21`, `30`, `35`, `67`, `71`, `133`, `139`, `154`, `204`, `212`, `273`, `303`, `321`, `322`, `358`, `377`, `409` ... (>30 distinct values)
- **`NK Cell Count`** (31 distinct): `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `11`, `12`, `15`, `17`, `21`, `22`, `23`, `24`, `31`, `35`, `43`, `49`, `63`, `75`, `108`, `142`, `151`, `164`, `168`, `171` ... (>30 distinct values)
- **`Total Immune Cells`** (31 distinct): `0`, `8`, `12`, `24`, `84`, `94`, `103`, `205`, `208`, `235`, `281`, `286`, `346`, `348`, `396`, `429`, `451`, `457`, `462`, `468`, `474`, `490`, `540`, `542`, `604`, `605`, `626`, `642`, `672`, `674` ... (>30 distinct values)
- **`GSIR SI`** (31 distinct): `0.0`, `0.11`, `0.16`, `0.23`, `0.25`, `0.26`, `0.3`, `0.32`, `0.36`, `0.41`, `0.43`, `0.45`, `0.46`, `0.48`, `0.52`, `0.58`, `0.59`, `0.6`, `0.65`, `0.66`, `0.68`, `0.7`, `0.72`, `0.74`, `0.77`, `0.78`, `0.83`, `0.84`, `0.88`, `0.91` ... (>30 distinct values)

## Database: `modalities`

20 tables.

### `BCR-seq` (320 rows)

```
CREATE TABLE `BCR-seq` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Cell_Type` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Sample` VARCHAR(255),
  `Technical replicate` VARCHAR(255),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-030`, `HPAP-031`, `HPAP-032` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `BCR-seq`
- **`Cell_Type`** (1 distinct): `B cell`
- **`Tissue`** (1 distinct): `Spleen`
- **`Sample`** (18 distinct): `Replicate-1`, `Replicate-1-vial2`, `Replicate-10`, `Replicate-11`, `Replicate-12`, `Replicate-13`, `Replicate-14`, `Replicate-15`, `Replicate-16`, `Replicate-2`, `Replicate-2-vial2`, `Replicate-3`, `Replicate-4`, `Replicate-5`, `Replicate-6`, `Replicate-7`, `Replicate-8`, `Replicate-9`
- **`Technical replicate`** (2 distinct): `Technical-1`, `Technical-2`
- **`File`** (31 distinct): `HPAP-001_IgHbulk_Spleen_DNA_100ng_Replicate-1_2018-02-23_S1_L001_R1_001.fastq...`, `HPAP-001_IgHbulk_Spleen_DNA_100ng_Replicate-2_2018-02-23_S20_L001_R1_001.fast...`, `HPAP-001_IgHbulk_Spleen_DNA_100ng_Replicate-3_2018-02-23_S41_L001_R1_001.fast...`, `HPAP-001_IgHbulk_Spleen_DNA_100ng_Replicate-4_2018-02-23_S58_L001_R1_001.fast...`, `HPAP-002_IgHbulk_Spleen_DNA_100ng_Replicate-1_2018-02-23_S2_L001_R1_001.fastq...`, `HPAP-002_IgHbulk_Spleen_DNA_100ng_Replicate-2_2018-02-23_S21_L001_R1_001.fast...`, `HPAP-003_IgHbulk_Spleen_DNA_100ng_Replicate-1_2018-02-23_S3_L001_R1_001.fastq...`, `HPAP-003_IgHbulk_Spleen_DNA_100ng_Replicate-2_2018-02-23_S22_L001_R1_001.fast...`, `HPAP-003_IgHbulk_Spleen_DNA_100ng_Replicate-3_2018-02-23_S42_L001_R1_001.fast...`, `HPAP-003_IgHbulk_Spleen_DNA_100ng_Replicate-4_2018-02-23_S59_L001_R1_001.fast...`, `HPAP-004_IgHbulk_Spleen_DNA_100ng_Replicate-1_2018-02-23_S4_L001_R1_001.fastq...`, `HPAP-004_IgHbulk_Spleen_DNA_100ng_Replicate-2_2018-02-23_S23_L001_R1_001.fast...`, `HPAP-004_IgHbulk_Spleen_DNA_100ng_Replicate-3_2018-02-23_S43_L001_R1_001.fast...`, `HPAP-004_IgHbulk_Spleen_DNA_100ng_Replicate-4_2018-02-23_S60_L001_R1_001.fast...`, `HPAP-005_IgHbulk_Spleen_DNA_100ng_Replicate-1_2018-02-23_S5_L001_R1_001.fastq...`, `HPAP-005_IgHbulk_Spleen_DNA_100ng_Replicate-2_2018-02-23_S24_L001_R1_001.fast...`, `HPAP-005_IgHbulk_Spleen_DNA_100ng_Replicate-3_2018-02-23_S44_L001_R1_001.fast...`, `HPAP-005_IgHbulk_Spleen_DNA_100ng_Replicate-4_2018-02-23_S61_L001_R1_001.fast...`, `HPAP-006_IgHbulk_Spleen_DNA_100ng_Replicate-1_2018-02-23_S6_L001_R1_001.fastq...`, `HPAP-006_IgHbulk_Spleen_DNA_100ng_Replicate-2_2018-02-23_S25_L001_R1_001.fast...`, `HPAP-006_IgHbulk_Spleen_DNA_100ng_Replicate-3_2018-02-23_S45_L001_R1_001.fast...`, `HPAP-006_IgHbulk_Spleen_DNA_100ng_Replicate-4_2018-02-23_S62_L001_R1_001.fast...`, `HPAP-007_IgHbulk_Spleen_DNA_100ng_Replicate-1_2018-02-23_S7_L001_R1_001.fastq...`, `HPAP-007_IgHbulk_Spleen_DNA_100ng_Replicate-2_2018-02-23_S26_L001_R1_001.fast...`, `HPAP-007_IgHbulk_Spleen_DNA_100ng_Replicate-3_2018-02-23_S46_L001_R1_001.fast...`, `HPAP-007_IgHbulk_Spleen_DNA_100ng_Replicate-4_2018-02-23_S63_L001_R1_001.fast...`, `HPAP-008_IgHbulk_Spleen_DNA_100ng_Replicate-1_2018-02-23_S8_L001_R1_001.fastq...`, `HPAP-008_IgHbulk_Spleen_DNA_100ng_Replicate-2_2018-02-23_S27_L001_R1_001.fast...`, `HPAP-008_IgHbulk_Spleen_DNA_100ng_Replicate-3_2018-02-23_S47_L001_R1_001.fast...`, `HPAP-008_IgHbulk_Spleen_DNA_100ng_Replicate-4_2018-02-23_S64_L001_R1_001.fast...` ... (>30 distinct values)
- **`Source`** (1 distinct): `Luning Prak Lab`
- **`Contact`** (1 distinct): `Adil Mohammed`

### `Bulk_ATAC-seq` (49 rows)

```
CREATE TABLE `Bulk_ATAC-seq` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Cell_Type` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-003`, `HPAP-004`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-022`, `HPAP-024`, `HPAP-035`, `HPAP-036`, `HPAP-038`, `HPAP-040`, `HPAP-045`, `HPAP-047`, `HPAP-051`, `HPAP-052`, `HPAP-054`, `HPAP-062`, `HPAP-063`, `HPAP-064`, `HPAP-066`, `HPAP-067` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `Bulk ATAC-seq`
- **`Cell_Type`** (3 distinct): ``, `alpha`, `beta`
- **`Tissue`** (2 distinct): `Exocrine`, `Pancreas`
- **`File`** (1 distinct): `PancDB_ATAC-seq_metadata_2022-06-15.xlsx`
- **`Source`** (1 distinct): `Klaus Kaestner_Upenn`
- **`Contact`** (1 distinct): `Xinyu Bao`

### `Bulk_RNA-seq` (146 rows)

```
CREATE TABLE `Bulk_RNA-seq` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Cell_Type` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Run` VARCHAR(255),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-015`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-022`, `HPAP-026`, `HPAP-029`, `HPAP-035`, `HPAP-036`, `HPAP-038`, `HPAP-040`, `HPAP-045`, `HPAP-047`, `HPAP-049`, `HPAP-051`, `HPAP-052` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `Bulk RNA-seq`
- **`Cell_Type`** (3 distinct): ``, `alpha`, `beta`
- **`Tissue`** (2 distinct): ``, `Exocrine`
- **`Run`** (12 distinct): `FGC1612;FGC1639`, `FGC1721`, `FGC1721;FGC1736`, `FGC1721;FGC1736;FGC1752`, `FGC1721;FGC1752;FGC1736`, `FGC1721;FGC1752;FGC1761;FGC1736`, `FGC1752;FGC1761`, `FGC2009`, `FGC2142;FGC2142;FGC2144;FGC2155;FGC2155`, `FGC2144;FGC2142;FGC2142;FGC2155;FGC2155`, `FGC2144;FGC2155;FGC2155`, `FGC2365;FGC2365;FGC2386;FGC2386`
- **`File`** (31 distinct): `HPAP-001_beta_53632_FGC1612_s_3_ACTTGA.fastq.gz;HPAP-001_beta_53632_FGC1639_s...`, `HPAP-001_beta_55897_FGC1612_s_3_AATCCAGC.fastq.gz;HPAP-001_beta_55897_FGC1639...`, `HPAP-001_exocrine_107457_FGC2365_s_1_GGCAAGTT.fastq.gz;HPAP-001_exocrine_1074...`, `HPAP-001_mRNAseq_Alpha_1_fastq-data.fastq.gz;HPAP-001_mRNAseq_Alpha_2_fastq-d...`, `HPAP-002_alpha_65844_FGC1721_s_4_ATCACG.fastq.gz`, `HPAP-002_exocrine_107458_FGC2365_s_1_GATCTTGC.fastq.gz;HPAP-002_exocrine_1074...`, `HPAP-003_beta_65847_FGC1721_s_4_GATCAG.fastq.gz;HPAP-003_beta_65847_FGC1736_s...`, `HPAP-003_exocrine_107459_FGC2365_s_1_CAATGCGA.fastq.gz;HPAP-003_exocrine_1074...`, `HPAP-003_mRNAseq_Alpha_1_fastq-data.fastq.gz;HPAP-003_mRNAseq_Alpha_2_fastq-d...`, `HPAP-004_beta_65849_FGC1721_s_4_GGCTAC.fastq.gz`, `HPAP-004_exocrine_107460_FGC2365_s_1_GGTGTACA.fastq.gz;HPAP-004_exocrine_1074...`, `HPAP-004_mRNAseq_Alpha_1_fastq-data.fastq.gz`, `HPAP-005_beta_65851_FGC1721_s_4_GTTTCG.fastq.gz;HPAP-005_beta_65851_FGC1736_s...`, `HPAP-005_exocrine_107461_FGC2365_s_1_TAGGAGCT.fastq.gz;HPAP-005_exocrine_1074...`, `HPAP-006_beta_65853_FGC1721_s_4_GAGTGG.fastq.gz;HPAP-006_beta_65853_FGC1752_s...`, `HPAP-006_exocrine_107462_FGC2365_s_1_CGAATTGC.fastq.gz;HPAP-006_exocrine_1074...`, `HPAP-006_mRNAseq_Alpha_1_fastq-data.fastq.gz;HPAP-006_mRNAseq_Alpha_2_fastq-d...`, `HPAP-007_beta_65855_FGC1721_s_4_ATTCCT.fastq.gz;HPAP-007_beta_65855_FGC1752_s...`, `HPAP-007_exocrine_107463_FGC2365_s_1_GTCCTAAG.fastq.gz;HPAP-007_exocrine_1074...`, `HPAP-007_mRNAseq_Alpha_1_fastq-data.fastq.gz;HPAP-007_mRNAseq_Alpha_2_fastq-d...`, `HPAP-008_beta_65857_FGC1721_s_4_TGACCA.fastq.gz`, `HPAP-008_mRNAseq_Alpha_1_fastq-data.fastq.gz`, `HPAP-009_beta_65859_FGC1721_s_4_GCCAAT.fastq.gz;HPAP-009_beta_65859_FGC1736_s...`, `HPAP-009_exocrine_107464_FGC2365_s_1_CTTAGGAC.fastq.gz;HPAP-009_exocrine_1074...`, `HPAP-009_mRNAseq_Alpha_1_fastq-data.fastq.gz;HPAP-009_mRNAseq_Alpha_2_fastq-d...`, `HPAP-010_beta_65861_FGC1721_s_4_CTTGTA.fastq.gz`, `HPAP-010_exocrine_107465_FGC2365_s_1_TCCACGTT.fastq.gz;HPAP-010_exocrine_1074...`, `HPAP-010_mRNAseq_Alpha_1_fastq-data.fastq.gz;HPAP-010_mRNAseq_Alpha_2_fastq-d...`, `HPAP-011_alpha_66892_FGC1752_s_8_CGATGT.fastq.gz;HPAP-011_alpha_66892_FGC1761...`, `HPAP-011_beta_66893_FGC1752_s_8_TGACCA.fastq.gz;HPAP-011_beta_66893_FGC1761_s...` ... (>30 distinct values)
- **`Source`** (1 distinct): `Klaus Kaestner_Upenn`
- **`Contact`** (1 distinct): `Dongliang Leng`

### `CITE-seq_Protein` (44 rows)

```
CREATE TABLE `CITE-seq_Protein` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Region` VARCHAR(255),
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (18 distinct): `HPAP-012`, `HPAP-016`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-023`, `HPAP-026`, `HPAP-027`, `HPAP-032`, `HPAP-045`, `HPAP-048`, `HPAP-095`, `HPAP-098`, `HPAP-099`, `HPAP-102`, `HPAP-107`, `HPAP-110`, `HPAP-114`
- **`Data Modality`** (1 distinct): `CITE-seq Protein`
- **`Tissue`** (4 distinct): `Exocrine`, `Lymph_node`, `Mes/islet`, `Spleen`
- **`Region`** (3 distinct): ``, `Head`, `Tail`
- **`Source`** (1 distinct): `Michael Betts_Upenn`
- **`Contact`** (1 distinct): `Haoxuan Zeng`

### `CODEX` (137 rows)

```
CREATE TABLE `CODEX` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Region` VARCHAR(255),
  `File` TEXT,
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-007`, `HPAP-009`, `HPAP-011`, `HPAP-013`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-018`, `HPAP-023`, `HPAP-024`, `HPAP-027`, `HPAP-030`, `HPAP-037`, `HPAP-038`, `HPAP-040`, `HPAP-042`, `HPAP-044`, `HPAP-045`, `HPAP-046`, `HPAP-047`, `HPAP-048`, `HPAP-053`, `HPAP-055`, `HPAP-056`, `HPAP-058`, `HPAP-059`, `HPAP-060`, `HPAP-062`, `HPAP-064`, `HPAP-065` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `CODEX`
- **`Tissue`** (1 distinct): `Pancreas`
- **`Region`** (3 distinct): `Body`, `Head`, `Tail`
- **`File`** (31 distinct): `HPAP-007_CODEX_Tail-of-pancreas_OCT.qptiff`, `HPAP-009_CODEX_Head-of-pancreas_OCT.qptiff`, `HPAP-009_CODEX_Tail-of-pancreas_OCT.ome.tif`, `HPAP-011_CODEX_Body-of-pancreas_OCT.qptiff`, `HPAP-013_CODEX_Body-of-pancreas_OCT.ome.tif`, `HPAP-013_CODEX_Tail-of-pancreas_OCT.ome.tiff`, `HPAP-015_CODEX_Head-of-pancreas_OCT.ome.tif`, `HPAP-015_CODEX_Tail-of-pancreas_OCT.qptiff`, `HPAP-016_CODEX_Body-of-pancreas_OCT.ome.tif`, `HPAP-016_CODEX_Tail-of-pancreas_OCT.qptiff`, `HPAP-017_CODEX_Body-of-pancreas_OCT.ome.tif`, `HPAP-017_CODEX_Tail-of-pancreas_OCT.qptiff`, `HPAP-018_CODEX_Head-of-pancreas_OCT.ome.tif`, `HPAP-018_CODEX_Tail-of-pancreas_OCT.qptiff`, `HPAP-023_CODEX_Tail-of-pancreas_OCT.qptiff`, `HPAP-024_CODEX_Head-of-pancreas_OCT.qptiff`, `HPAP-024_CODEX_Tail-of-pancreas_OCT.ome.tif`, `HPAP-027_CODEX_Head-of-pancreas_OCT.ome.tif`, `HPAP-027_CODEX_Tail-of-pancreas_OCT.ome.tiff`, `HPAP-030_CODEX_Tail-of-pancreas_OCT.qptiff`, `HPAP-037_CODEX_Head-of-pancreas_OCT.ome.tif`, `HPAP-037_CODEX_Tail-of-pancreas_OCT.ome.tiff`, `HPAP-038_CODEX_Tail-of-pancreas_OCT.qptiff`, `HPAP-040_CODEX_Head-of-pancreas_OCT.ome.tif`, `HPAP-040_CODEX_Tail-of-pancreas_OCT.ome.tiff`, `HPAP-042_CODEX_Head-of-pancreas_OCT.ome.tif`, `HPAP-042_CODEX_Tail-of-pancreas_OCT.ome.tiff`, `HPAP-044_CODEX_Body-of-pancreas_OCT.ome.tif`, `HPAP-044_CODEX_Tail-of-pancreas_OCT.ome.tiff`, `HPAP-045_CODEX_Head-of-pancreas_OCT.qptiff` ... (>30 distinct values)
- **`Contact`** (1 distinct): `Feng Fan`

### `Calcium_Imaging` (724 rows)

```
CREATE TABLE `Calcium_Imaging` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Region` VARCHAR(255),
  `Run` VARCHAR(255),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-014`, `HPAP-015`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-027`, `HPAP-029`, `HPAP-034`, `HPAP-035`, `HPAP-036`, `HPAP-038`, `HPAP-043`, `HPAP-045`, `HPAP-047`, `HPAP-049`, `HPAP-050`, `HPAP-051`, `HPAP-052`, `HPAP-053`, `HPAP-054`, `HPAP-055`, `HPAP-056`, `HPAP-057`, `HPAP-058` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `Calcium Imaging`
- **`Tissue`** (1 distinct): `Islet`
- **`Region`** (12 distinct): `Region 1`, `Region 10`, `Region 11`, `Region 12`, `Region 2`, `Region 3`, `Region 4`, `Region 5`, `Region 6`, `Region 7`, `Region 8`, `Region 9`
- **`Run`** (4 distinct): `Run1`, `Run2`, `Run3`, `Run4`
- **`File`** (31 distinct): `HPAP-002_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-003_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-004_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-014_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-014_Calcium-imaging-Whole-islets_Run2_data.xlsx`, `HPAP-014_Calcium-imaging-Whole-islets_Run3_data.xlsx`, `HPAP-014_Calcium-imaging-Whole-islets_Run4_data.xlsx`, `HPAP-015_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-015_Calcium-imaging-Whole-islets_Run2_data.xlsx`, `HPAP-015_Calcium-imaging-Whole-islets_Run3_data.xlsx`, `HPAP-015_Calcium-imaging-Whole-islets_Run4_data.xlsx`, `HPAP-017_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-017_Calcium-imaging-Whole-islets_Run2_data.xlsx`, `HPAP-017_Calcium-imaging-Whole-islets_Run3_data.xlsx`, `HPAP-017_Calcium-imaging-Whole-islets_Run4_data.xlsx`, `HPAP-018_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-018_Calcium-imaging-Whole-islets_Run2_data.xlsx`, `HPAP-018_Calcium-imaging-Whole-islets_Run3_data.xlsx`, `HPAP-018_Calcium-imaging-Whole-islets_Run4_data.xlsx`, `HPAP-019_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-019_Calcium-imaging-Whole-islets_Run2_data.xlsx`, `HPAP-019_Calcium-imaging-Whole-islets_Run3_data.xlsx`, `HPAP-019_Calcium-imaging-Whole-islets_Run4_data.xlsx`, `HPAP-020_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-020_Calcium-imaging-Whole-islets_Run2_data.xlsx`, `HPAP-020_Calcium-imaging-Whole-islets_Run3_data.xlsx`, `HPAP-021_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-022_Calcium-imaging-Whole-islets_Run1_data.xlsx`, `HPAP-022_Calcium-imaging-Whole-islets_Run2_data.xlsx`, `HPAP-022_Calcium-imaging-Whole-islets_Run3_data.xlsx` ... (>30 distinct values)
- **`Source`** (1 distinct): `Stoffers Lab_Upenn`
- **`Contact`** (1 distinct): `Jeya Vandana`

### `CyTOF` (142 rows)

```
CREATE TABLE `CyTOF` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Markers` TEXT,
  `File` TEXT,
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-032`, `HPAP-033` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `CyTOF`
- **`Tissue`** (1 distinct): `Islet`
- **`Markers`** (6 distinct): `52`, `53`, `55`, `62`, `66`, `67`
- **`File`** (31 distinct): `HPAP-001_CyTOF_data.fcs`, `HPAP-002_CyTOF_data.fcs`, `HPAP-003_CyTOF_data.fcs`, `HPAP-004_CyTOF_data.fcs`, `HPAP-005_CyTOF_data.fcs`, `HPAP-006_CyTOF_data.fcs`, `HPAP-007_CyTOF_data.fcs`, `HPAP-008_CyTOF_data.fcs`, `HPAP-009_CyTOF_data.fcs`, `HPAP-010_CyTOF_data.fcs`, `HPAP-011_CyTOF_data.fcs`, `HPAP-012_CyTOF_data.fcs`, `HPAP-013_CyTOF_data.fcs`, `HPAP-014_CyTOF_data.fcs`, `HPAP-015_CyTOF_data.fcs`, `HPAP-016_CyTOF_data.fcs`, `HPAP-017_CyTOF_data.fcs`, `HPAP-018_CyTOF_data.fcs`, `HPAP-019_CyTOF_data.fcs`, `HPAP-020_CyTOF_data.fcs`, `HPAP-021_CyTOF_data.fcs`, `HPAP-022_CyTOF_data.fcs`, `HPAP-023_CyTOF_data.fcs`, `HPAP-024_CyTOF_data.fcs`, `HPAP-026_CyTOF_data.fcs`, `HPAP-027_CyTOF_data.fcs`, `HPAP-028_CyTOF_data.fcs`, `HPAP-029_CyTOF_data.fcs`, `HPAP-032_CyTOF_data.fcs`, `HPAP-033_CyTOF_data.fcs` ... (>30 distinct values)
- **`Contact`** (1 distinct): `Dongliang Leng`

### `Data_Track` (21 rows)

```
CREATE TABLE `Data_Track` (
  `Data Modality` VARCHAR(255),
  `Documents` TEXT,
  `Pipeline` TEXT,
  `After QC Data` TEXT,
  `Metadata` TEXT,
  `Other / Notes` TEXT,
  `Storage Location (primary)` TEXT,
  `Responsible Person` VARCHAR(255),
  `Email` VARCHAR(255),
  `Data Status` VARCHAR(255),
  `Unnamed: 10` TEXT
);
```

**Column values:**

- **`Data Modality`** (21 distinct): `BCR-seq`, `Bulk ATAC-seq`, `Bulk RNA-seq`, `Bulk WGBS (DNA Methylome)`, `Calcium-Imaging`, `CITE-seq`, `CODEX`, `CyTOF`, `Flow Cytometry`, `Functional Islet Perifusion`, `Histology(Chen lab)`, `Histology(HPAP)`, `Imaging Mass Cytometry`, `Oxygen Consumption`, `Patch-seq`, `scATAC-seq`, `scRNA-seq`, `Single-cell Multiome`, `TCR-seq`, `TEA-seq (ATAC+RNA+ADT)`, `WGS`
- **`Documents`** (7 distinct): ``, `[Available] Google Doc protocol:
docs.google.com/…/1BucDmbW4cE…`, `[Available] Google Doc protocol:
docs.google.com/…/1goNXlIK6O4j…`, `[Available] Google Doc protocol:
docs.google.com/…/1oZfpNfuDidLho5k…`, `[Available] H&E Axioscan: raw .czi (Slide 1: 01.czi, Slide 2: 02 rescan.czi) ...`, `[Missing]`, `/umms-drjieliu/proj/MAI_T1Ddata/Patchseq/`
- **`Pipeline`** (11 distinct): `(not needed)`, `[Available] (same Google Doc includes pipeline)`, `[Available] bodenmillergroup.github.io/
ImcSegmentationPipeline
+ GitHub: Bod...`, `[Available] Drive folder (pipeline docs)`, `[Available] Drive:
drive.google.com/…/1GX2GrBNQ0v…`, `[Available] FastQC + MiXCR pipeline
(github.com/s-andrews/FastQC;
mixcr.com)`, `[Available] FastQC + MiXCR pipeline
(same as BCR-seq)`, `[Available] github.com/PanKbase/HPAP-scATAC-seq
[Partial / Needs Verification...`, `[Missing]`, `/nfs/turbo/umms-drjieliu/usr/xinyubao/ATACseq-NextFlow`, `/umms-drjieliu/proj/MAI_T1Ddata/Patchseq/`
- **`After QC Data`** (15 distinct): ``, `[Available] /nfs/turbo/umms-drjieliu/usr/dongleng/
01.Bulk_RNA.seq…/04.vst_ma...`, `[Available] /nfs/turbo/umms-drjieliu/usr/dongleng/
02.TCR_BCR.Adil`, `[Available] /nfs/turbo/umms-drjieliu/usr/dongleng/02.Flow_cytometery.T1D/05.F...`, `[Available] Cornell Box (see Storage Location): .czi images + Project.qpproj`, `[Available] drive.google.com + /nfs/turbo/…
FunctionPerifusion_Vanderbilt/`, `[Available] drive.google.com/…/1h2ghyIQA3t…`, `[Available] drive.google.com/…/1RK45VTtT0A…`, `[Missing]`, `[Missing]
(Processing data path only:
/FM_diabetes/data/scATAC_RNA_pankbase;
...`, `[Partial / Needs Verification] /nfs/turbo/umms-drjieliu/usr/luosanj/
FM_diabe...`, `[Partial / Needs Verification] Partial — RNA:
/umms-drjieliu/proj/MAI_T1D_Dat...`, `/nfs/turbo/umms-drjieliu/usr/xinyubao/ATACseq-NextFlow`, `/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/export`, `/umms-drjieliu/proj/MAI_T1Ddata/Patchseq/`
- **`Metadata`** (10 distinct): `[Available] ADT / protein marker list:
/nfs/turbo/umms-drjieliu/proj/MAI_T1Dd...`, `[Available] Cell type annotation summary (per tissue/donor):
/nfs/turbo/umms-...`, `[Available] QC + Raw metadata (Google Sheets)`, `[Available] QC metadata (Google Sheets)`, `[Available] QC metadata + Raw metadata
(Google Sheets links)`, `[Missing]`, `/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/export_metadata`, `/umms-drjieliu/proj/MAI_T1Ddata/Patchseq/`, `Cleaned_PancDB_ATAC-seq_metadata_2022-06-15_v2.xlsx`, `Histology Metadata`
- **`Other / Notes`** (14 distinct): `—`, `[Missing]`, `[Partial / Needs Verification] Pipeline GitHub link was filled in the wrong f...`, `Last updated: 11/01`, `Last updated: 12/01`, `No pipeline or metadata submitted;
cell type annotations present in one path`, `Pancreas B/T sub-datasets also tracked (rows 9-10 in Form)`, `QC notes: HPAP-148 borderline (BCR);
HPAP-148, HPAP-154 borderline (TCR)`, `QC-Passed Data col (col 5) is EMPTY in form.
Text update in form rows 20-22 m...`, `Shared submission with BCR-seq (row 8 in Form)`, `v1; last updated 02/05/2026`, `v1; last updated 11/11/2025`, `v1; last updated 11/12/2025;
additional Vanderbilt data (contact: fengfan)`, `v1; last updated 12/01/2025`
- **`Storage Location (primary)`** (13 distinct): ``, `—`, `/nfs/turbo/umms-drjieliu/proj/MAI_T1D_Data/snMultiome/
(multiple sub-paths)`, `/nfs/turbo/umms-drjieliu/usr/dongleng/
01.Bulk_RNA.seq.for_T1D_immno_model/`, `/nfs/turbo/umms-drjieliu/usr/dongleng/
02.TCR_BCR.Adil`, `/nfs/turbo/umms-drjieliu/usr/luosanj/
FM_diabetes/data/CODEX/

/nfs/turbo/umm...`, `/nfs/turbo/umms-drjieliu/usr/luosanj/
FM_diabetes/data/scATAC_RNA_pankbase`, `/umms-drjieliu/usr/xinyubao/ATACseq-NextFlow/results/`, `Google Drive + /nfs/turbo/umms-drjieliu/proj/
MAI_T1D_Data/FunctionPerifusion...`, `Google Drive folder:
drive.google.com/…/1RvVeC0pvDGv…`, `Google Drive folder:
drive.google.com/…/1um8LIqqMVqN…`, `https://docs.google.com/spreadsheets/d/124D9N5GJQdkOBiyLlvYq6uZ4vJHmhKqL7bUCo...`, `https://wcm.box.com/s/o6nv0b0w4nityjugkh7ctp1wc7mkljwo`
- **`Responsible Person`** (10 distinct): `—`, `— (name not provided)`, `Adil`, `Dongliang Leng`, `Jeya`, `Jeya + fengfan`, `Sally Lee`, `See Email column (two contacts from form)`, `Xinyu Bao`, `Yiicheng Tao`
- **`Email`** (8 distinct): ``, `—`, `aim4007@med.cornell.edu; am2832@cornell.edu`, `dol4005@med.cornell.edu`, `jeyavandana@gmail.com`, `sl2767@cornell.edu`, `xinyubao@umich.edu `, `yctao@umich.edu（not）`
- **`Data Status`** (10 distinct): `[Available] Available`, `[Missing] Not Submitted`, `[Partial / Needs Verification] Documents Missing`, `[Partial / Needs Verification] metadata  Missing`, `[Partial / Needs Verification] Metadata & Documents Missing`, `[Partial / Needs Verification] Only Pipeline Submitted — Data & Metadata Missing`, `[Partial / Needs Verification] Partial — Pipeline & Metadata & Documents Missing`, `[Partial / Needs Verification] Partial — Pipeline & Metadata Missing`, `[Partial / Needs Verification] Partial — QC Data & Metadata & Documents Missi...`, `[Partial / Needs Verification] Pipeline & QC Data & Metadata Missing
(QC data...`
- **`Unnamed: 10`** (2 distinct): ``, ` detailed cell masks in GeoJson format.`

### `Flow_Cytometry` (1691 rows)

```
CREATE TABLE `Flow_Cytometry` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Cell_Type` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-025`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-030`, `HPAP-031`, `HPAP-032` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `Flow Cytometry`
- **`Cell_Type`** (3 distinct): `B cell`, `Immune_lineage`, `T cell`
- **`Tissue`** (3 distinct): `Lymph_node`, `PBMC`, `Spleen`
- **`File`** (31 distinct): `HPAP_001_Flow_Bc_Spleen_data.fcs`, `HPAP_001_Flow_Lineage_Spleen_data.fcs`, `HPAP_001_Flow_TetramerMixA_Lymph_node_body_of_pancreas_data.fcs`, `HPAP_001_Flow_TetramerMixA_Lymph_node_head_of_pancreas_data.fcs`, `HPAP_001_Flow_TetramerMixA_Lymph_node_tail_of_pancreas_data.fcs`, `HPAP_001_Flow_TetramerMixB_Lymph_node_body_of_pancreas_data.fcs`, `HPAP_001_Flow_TetramerMixB_Lymph_node_head_of_pancreas_data.fcs`, `HPAP_001_Flow_TetramerMixB_Lymph_node_tail_of_pancreas_data.fcs`, `HPAP_003_Flow_Bc_Spleen_data.fcs`, `HPAP_003_Flow_Lineage_Spleen_data.fcs`, `HPAP_003_Flow_TetramerMixA_Lymph_node_body_of_pancreas_data.fcs`, `HPAP_003_Flow_TetramerMixA_Lymph_node_head_of_pancreas_data.fcs`, `HPAP_003_Flow_TetramerMixA_Lymph_node_mesentery_data.fcs`, `HPAP_003_Flow_TetramerMixA_PBMC_data.fcs`, `HPAP_003_Flow_TetramerMixA_Spleen_data.fcs`, `HPAP_003_Flow_TetramerMixB_Lymph_node_body_of_pancreas_data.fcs`, `HPAP_003_Flow_TetramerMixB_Lymph_node_head_of_pancreas_data.fcs`, `HPAP_003_Flow_TetramerMixB_Lymph_node_mesentery_data.fcs`, `HPAP_003_Flow_TetramerMixB_PBMC_data.fcs`, `HPAP_003_Flow_TetramerMixB_Spleen_data.fcs`, `HPAP_004_Flow_Bc_Spleen_data.fcs`, `HPAP_004_Flow_Lineage_Spleen_data.fcs`, `HPAP_005_Flow_Bc_Spleen_data.fcs`, `HPAP_005_Flow_Lineage_Spleen_data.fcs`, `HPAP_005_Flow_TetramerMixA_Lymph_node_mesentery_data.fcs`, `HPAP_005_Flow_TetramerMixA_PBMC_data.fcs`, `HPAP_005_Flow_TetramerMixB_Lymph_node_mesentery_data.fcs`, `HPAP_005_Flow_TetramerMixB_PBMC_data.fcs`, `HPAP_006_Flow_Bc_Spleen_data.fcs`, `HPAP_006_Flow_Lineage_Spleen_data.fcs` ... (>30 distinct values)
- **`Source`** (1 distinct): `Kaestner Lab_Upenn`
- **`Contact`** (1 distinct): `Dongliang Leng`

### `Histology` (708 rows)

```
CREATE TABLE `Histology` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `svs_total` DOUBLE,
  `svs_ffpe` DOUBLE,
  `svs_oct_flash_frozen` DOUBLE,
  `svs_oct_lightly_fixed` DOUBLE,
  `svs_other` DOUBLE,
  `ndpi_total` DOUBLE,
  `ndpi_ffpe` DOUBLE,
  `ndpi_oct_flash_frozen` DOUBLE,
  `ndpi_oct_lightly_fixed` DOUBLE,
  `ndpi_other` DOUBLE,
  `locations` TEXT,
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-025`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-030` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `Histology`
- **`Tissue`** (6 distinct): `Artery`, `Duodenum`, `Lymph_node`, `Pancreas`, `Spleen`, `Thymus`
- **`svs_total`** (20 distinct): `0.0`, `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `7.0`, `8.0`, `9.0`, `10.0`, `11.0`, `12.0`, `13.0`, `14.0`, `15.0`, `17.0`, `18.0`, `19.0`, `23.0`
- **`svs_ffpe`** (11 distinct): `0.0`, `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `7.0`, `9.0`, `10.0`, `14.0`
- **`svs_oct_flash_frozen`** (11 distinct): `0.0`, `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `7.0`, `8.0`, `9.0`, `12.0`
- **`svs_oct_lightly_fixed`** (5 distinct): `0.0`, `1.0`, `2.0`, `3.0`, `6.0`
- **`svs_other`** (1 distinct): `0.0`
- **`ndpi_total`** (15 distinct): `0.0`, `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `7.0`, `8.0`, `9.0`, `10.0`, `11.0`, `12.0`, `13.0`, `14.0`
- **`ndpi_ffpe`** (15 distinct): `0.0`, `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `7.0`, `8.0`, `9.0`, `10.0`, `11.0`, `12.0`, `13.0`, `14.0`
- **`ndpi_oct_flash_frozen`** (7 distinct): `0.0`, `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `6.0`
- **`ndpi_oct_lightly_fixed`** (3 distinct): `0.0`, `1.0`, `3.0`
- **`ndpi_other`** (2 distinct): `0.0`, `4.0`
- **`locations`** (29 distinct): ``, `Artery`, `Body-of-pancreas`, `Body-of-pancreas | Head-of-pancreas | Tail-of-pancreas`, `Duodenum`, `Duodenum | Duodenum-only`, `Duodenum-distal-one-third | Duodenum-mid-one-third | Duodenum-proximal-one-third`, `Duodenum-only`, `Duodenum-proximal-one-third`, `Duodenum-Proximal-one-third | Duodenum-only`, `Duodenum-unsure-of-orientation`, `Lymph-node`, `Lymph-node-Body-of-pancreas | Lymph-node-Head-of-pancreas`, `Lymph-node-body-of-pancreas | Lymph-node-head-of-pancreas | Lymph-node-mesent...`, `Lymph-node-body-of-pancreas | Lymph-node-head-of-pancreas | Lymph-node-mesent...`, `Lymph-node-Body-of-pancreas | Lymph-node-Head-of-pancreas | Lymph-node-SMA`, `Lymph-node-body-of-pancreas | Lymph-node-head-of-pancreas | Lymph-node-sma | ...`, `Lymph-node-Body-of-pancreas | Lymph-node-Head-of-pancreas | Lymph-node-Tail-o...`, `Lymph-node-Body-of-pancreas | Lymph-node-SMA | Lymph-node-Tail-of-pancreas`, `Lymph-node-Body-of-pancreas | Lymph-node-Tail-of-pancreas`, `Lymph-node-Head-of-pancreas`, `Lymph-node-Head-of-pancreas | Lymph-node-SMA`, `Lymph-node-Head-of-pancreas | Lymph-node-SMA | Lymph-node-Tail-of-pancreas`, `Lymph-node-Head-of-pancreas | Lymph-node-Tail-of-pancreas`, `Lymph-node-sma | Lymph-node-tail-of-pancreas`, `Lymph-node-tail-of-pancreas`, `Pancreas-unsure-of-orientation`, `Spleen`, `Thymus`
- **`Contact`** (1 distinct): `Adil Mohammed`

### `IMC` (87 rows)

```
CREATE TABLE `IMC` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-007`, `HPAP-009`, `HPAP-011`, `HPAP-013`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-018`, `HPAP-023`, `HPAP-024`, `HPAP-027`, `HPAP-030`, `HPAP-037`, `HPAP-038`, `HPAP-040`, `HPAP-042`, `HPAP-044`, `HPAP-045`, `HPAP-046`, `HPAP-047`, `HPAP-048`, `HPAP-053`, `HPAP-055`, `HPAP-056`, `HPAP-058`, `HPAP-059`, `HPAP-060`, `HPAP-062`, `HPAP-064`, `HPAP-065` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `IMC`
- **`Tissue`** (1 distinct): `Pancreas`
- **`Source`** (1 distinct): `Vanderbilt University Medical Center`
- **`Contact`** (1 distinct): `Yicheng Tao`

### `Overview` (17 rows)

```
CREATE TABLE `Overview` (
  `Data Modality` VARCHAR(255),
  `Category` VARCHAR(255),
  `Contact` VARCHAR(255),
  `Donors` BIGINT,
  `Records` BIGINT
);
```

**Column values:**

- **`Data Modality`** (17 distinct): `BCR-seq`, `Bulk ATAC-seq`, `Bulk RNA-seq`, `Calcium Imaging`, `CITE-seq Protein`, `CODEX`, `CyTOF`, `Flow Cytometry`, `Histology`, `IMC`, `Oxygen Consumption`, `Patch-seq`, `Perifusion`, `scATAC-seq`, `scRNA-seq`, `snMultiomics`, `TCR-seq`
- **`Category`** (5 distinct): `Cytometry`, `Functional`, `Imaging`, `Immune Repertoire`, `Sequencing`
- **`Contact`** (9 distinct): `Adil Mohammed`, `Dongliang Leng`, `Feng Fan`, `Haoxuan Zeng`, `Jeya Vandana`, `Jeya Vandana; Feng Fan`, `PanKbase/Kai Liu`, `Xinyu Bao`, `Yicheng Tao`
- **`Donors`** (16 distinct): `18`, `23`, `33`, `47`, `64`, `75`, `76`, `78`, `87`, `89`, `91`, `93`, `95`, `126`, `142`, `184`
- **`Records`** (16 distinct): `44`, `47`, `49`, `79`, `87`, `106`, `137`, `142`, `146`, `190`, `262`, `320`, `708`, `724`, `1173`, `1691`

### `Oxygen_Consumption` (87 rows)

```
CREATE TABLE `Oxygen_Consumption` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-026`, `HPAP-027`, `HPAP-029`, `HPAP-034`, `HPAP-035`, `HPAP-036`, `HPAP-038`, `HPAP-040`, `HPAP-042`, `HPAP-043`, `HPAP-045` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `Oxygen Consumption`
- **`Tissue`** (1 distinct): `Islet`
- **`File`** (31 distinct): `HPAP-001_Oxygen-consumption_data.xlsx`, `HPAP-002_Oxygen-consumption_data.xlsx`, `HPAP-003_Oxygen-consumption_data.xlsx`, `HPAP-004_Oxygen-consumption_data.xlsx`, `HPAP-005_Oxygen-consumption_data.xlsx`, `HPAP-006_Oxygen-consumption_data.xlsx`, `HPAP-007_Oxygen-consumption_data.xlsx`, `HPAP-008_Oxygen-consumption_data.xlsx`, `HPAP-010_Oxygen-consumption_data.xlsx`, `HPAP-011_Oxygen-consumption_data.xlsx`, `HPAP-012_Oxygen-consumption_data.xlsx`, `HPAP-013_Oxygen-consumption_data.xlsx`, `HPAP-014_Oxygen-consumption_data.xlsx`, `HPAP-018_Oxygen-consumption_data.xlsx`, `HPAP-019_Oxygen-consumption_data.xlsx`, `HPAP-020_Oxygen-consumption_data.xlsx`, `HPAP-021_Oxygen-consumption_data.xlsx`, `HPAP-022_Oxygen-consumption_data.xlsx`, `HPAP-023_Oxygen-consumption_data.xlsx`, `HPAP-026_Oxygen-consumption_data.xlsx`, `HPAP-027_Oxygen-consumption_data.xlsx`, `HPAP-029_Oxygen-consumption_data.xlsx`, `HPAP-034_Oxygen-consumption_data.xlsx`, `HPAP-035_Oxygen-consumption_data.xlsx`, `HPAP-036_Oxygen-consumption_data.xlsx`, `HPAP-038_Oxygen-consumption_data.xlsx`, `HPAP-040_Oxygen-consumption_data.xlsx`, `HPAP-042_Oxygen-consumption_data.xlsx`, `HPAP-043_Oxygen-consumption_data.xlsx`, `HPAP-045_Oxygen-consumption_data.xlsx` ... (>30 distinct values)
- **`Source`** (1 distinct): `Stoffers Lab_Upenn`
- **`Contact`** (1 distinct): `Jeya Vandana`

### `Patch-seq` (1173 rows)

```
CREATE TABLE `Patch-seq` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Cell_Type` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `plate` VARCHAR(100),
  `well` VARCHAR(100),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (23 distinct): `HPAP-051`, `HPAP-052`, `HPAP-053`, `HPAP-054`, `HPAP-057`, `HPAP-058`, `HPAP-059`, `HPAP-065`, `HPAP-066`, `HPAP-074`, `HPAP-075`, `HPAP-077`, `HPAP-079`, `HPAP-081`, `HPAP-083`, `HPAP-090`, `HPAP-091`, `HPAP-093`, `HPAP-096`, `HPAP-097`, `HPAP-105`, `HPAP-106`, `HPAP-108`
- **`Data Modality`** (1 distinct): `Patch-seq`
- **`Cell_Type`** (7 distinct): `acinar`, `alpha`, `beta`, `delta`, `ductal`, `INS+GCG+`, `PP`
- **`Tissue`** (1 distinct): `Islet`
- **`plate`** (19 distinct): `A`, `B`, `C`, `D`, `E`, `F`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`
- **`well`** (31 distinct): `10A`, `10B`, `10C`, `10D`, `10E`, `10F`, `10G`, `10H`, `11A`, `11B`, `11C`, `11D`, `11E`, `11F`, `11G`, `11H`, `12A`, `12B`, `12C`, `12D`, `12E`, `12F`, `12G`, `12H`, `1A`, `1B`, `1C`, `1D`, `1E`, `1F` ... (>30 distinct values)
- **`File`** (31 distinct): `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...`, `/athena/chenlab/scratch/jjv4001/Patchseq/processedpatchnew/HPAP-051_patchseq_...` ... (>30 distinct values)
- **`Source`** (1 distinct): `Macdonald Lab_Ualberta`
- **`Contact`** (1 distinct): `Jeya Vandana`

### `Perifusion` (190 rows)

```
CREATE TABLE `Perifusion` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Region` VARCHAR(255),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-002`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-016`, `HPAP-017`, `HPAP-018`, `HPAP-019`, `HPAP-020`, `HPAP-022`, `HPAP-024`, `HPAP-026`, `HPAP-027`, `HPAP-029`, `HPAP-032`, `HPAP-034`, `HPAP-035`, `HPAP-036`, `HPAP-037`, `HPAP-038` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `Perifusion`
- **`Tissue`** (1 distinct): `Islet`
- **`Region`** (5 distinct): ``, `Body`, `Head`, `Tail`, `Tali`
- **`File`** (31 distinct): ``, `HPAP-001_Perifusion_data.csv`, `HPAP-002_Perifusion_data.csv`, `HPAP-003_Perifusion_data.csv`, `HPAP-004_Perifusion_data.csv`, `HPAP-005_Perifusion_data.csv`, `HPAP-006_Perifusion_data.csv`, `HPAP-007_Perifusion_data.csv`, `HPAP-008_Perifusion_data.csv`, `HPAP-009_Perifusion_data.csv`, `HPAP-010_Perifusion_data.csv`, `HPAP-011_Perifusion_data.csv`, `HPAP-012_Perifusion_data.csv`, `HPAP-013_Perifusion_data.csv`, `HPAP-014_Perifusion_data.csv`, `HPAP-016_Perifusion_data.csv`, `HPAP-017_Perifusion_data.csv`, `HPAP-018_Perifusion_data.csv`, `HPAP-019_Perifusion_data.csv`, `HPAP-020_Perifusion_data.csv`, `HPAP-022_Perifusion_data.csv`, `HPAP-024_Perifusion_data.csv`, `HPAP-026_Perifusion_data.csv`, `HPAP-027_Perifusion_data.csv`, `HPAP-029_Perifusion_data.csv`, `HPAP-034_Perifusion_data.csv`, `HPAP-035_Perifusion_data.csv`, `HPAP-036_Perifusion_data.csv`, `HPAP-037_Perifusion_data.csv`, `HPAP-038_Perifusion_data.csv` ... (>30 distinct values)
- **`Source`** (2 distinct): ``, `Stoffers Lab_Upenn`
- **`Contact`** (2 distinct): `Feng Fan`, `Jeya Vandana`

### `Summary` (11 rows)

```
CREATE TABLE `Summary` (
  `Tissue / Data Modality` VARCHAR(255),
  `Bulk ATAC-seq` DOUBLE,
  `Bulk RNA-seq` DOUBLE,
  `CITE-seq Protein` DOUBLE,
  `Flow Cytometry` DOUBLE,
  `snMultiomics` DOUBLE,
  `scATAC-seq` DOUBLE,
  `scRNA-seq` DOUBLE,
  `IMC` DOUBLE,
  `BCR-seq` DOUBLE,
  `TCR-seq` DOUBLE,
  `Perifusion` DOUBLE,
  `Histology` DOUBLE,
  `CyTOF` DOUBLE,
  `CODEX` DOUBLE,
  `Calcium Imaging` DOUBLE,
  `Patch-seq` DOUBLE,
  `Oxygen Consumption` DOUBLE
);
```

**Column values:**

- **`Tissue / Data Modality`** (11 distinct): `Artery`, `Duodenum`, `Exocrine (Pancreas)`, `Islet (Pancreas)`, `Lymph Node (PLN)`, `Mes/Islet`, `Pancreas (whole)`, `PBMC`, `Spleen`, `Thymus`, `Total (unique donors)`
- **`Bulk ATAC-seq`** (4 distinct): `0.0`, `4.0`, `31.0`, `35.0`
- **`Bulk RNA-seq`** (2 distinct): `0.0`, `45.0`
- **`CITE-seq Protein`** (6 distinct): `0.0`, `3.0`, `10.0`, `14.0`, `17.0`, `44.0`
- **`Flow Cytometry`** (5 distinct): `0.0`, `72.0`, `79.0`, `88.0`, `239.0`
- **`snMultiomics`** (6 distinct): `0.0`, `2.0`, `20.0`, `29.0`, `48.0`, `99.0`
- **`scATAC-seq`** (2 distinct): `0.0`, `47.0`
- **`scRNA-seq`** (2 distinct): `0.0`, `78.0`
- **`IMC`** (2 distinct): `0.0`, `87.0`
- **`BCR-seq`** (2 distinct): `0.0`, `95.0`
- **`TCR-seq`** (2 distinct): `0.0`, `93.0`
- **`Perifusion`** (2 distinct): `0.0`, `126.0`
- **`Histology`** (8 distinct): `0.0`, `1.0`, `3.0`, `168.0`, `177.0`, `178.0`, `181.0`, `708.0`
- **`CyTOF`** (2 distinct): `0.0`, `142.0`
- **`CODEX`** (2 distinct): `0.0`, `91.0`
- **`Calcium Imaging`** (2 distinct): `0.0`, `76.0`
- **`Patch-seq`** (2 distinct): `0.0`, `23.0`
- **`Oxygen Consumption`** (2 distinct): `0.0`, `87.0`

### `TCR-seq` (262 rows)

```
CREATE TABLE `TCR-seq` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Cell_Type` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `File` TEXT,
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-001`, `HPAP-003`, `HPAP-004`, `HPAP-005`, `HPAP-006`, `HPAP-007`, `HPAP-008`, `HPAP-009`, `HPAP-010`, `HPAP-011`, `HPAP-012`, `HPAP-013`, `HPAP-014`, `HPAP-015`, `HPAP-016`, `HPAP-017`, `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-026`, `HPAP-027`, `HPAP-028`, `HPAP-029`, `HPAP-030`, `HPAP-031`, `HPAP-032`, `HPAP-033` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `TCR-seq`
- **`Cell_Type`** (1 distinct): `T cell`
- **`Tissue`** (1 distinct): `Spleen`
- **`File`** (31 distinct): `HPAP-001_TCRbulk_Spleen_DNA_100ng_Replicate-1_2021-10-22_S1_L001_R1_001.fastq...`, `HPAP-001_TCRbulk_Spleen_DNA_100ng_Replicate-2_2021-10-22_S2_L001_R1_001.fastq...`, `HPAP-001_TCRbulk_Spleen_DNA_100p0ng_Replicate-1_2021-11-15_S1_L001_R1_001.fas...`, `HPAP-001_TCRbulk_Spleen_DNA_100p0ng_Replicate-2_2021-11-15_S2_L001_R1_001.fas...`, `HPAP-003_TCRbulk_Spleen_DNA_100ng_Replicate-1_2021-10-22_S62_L001_R1_001.fast...`, `HPAP-003_TCRbulk_Spleen_DNA_100ng_Replicate-2_2021-10-22_S63_L001_R1_001.fast...`, `HPAP-003_TCRbulk_Spleen_DNA_100p0ng_Replicate-1_2021-11-15_S62_L001_R1_001.fa...`, `HPAP-003_TCRbulk_Spleen_DNA_100p0ng_Replicate-2_2021-11-15_S63_L001_R1_001.fa...`, `HPAP-004_TCRbulk_Spleen_DNA_100ng_Replicate-1_2021-10-22_S64_L001_R2_001.fast...`, `HPAP-004_TCRbulk_Spleen_DNA_100ng_Replicate-2_2021-10-22_S65_L001_R1_001.fast...`, `HPAP-004_TCRbulk_Spleen_DNA_100p0ng_Replicate-1_2021-11-15_S64_L001_R1_001.fa...`, `HPAP-004_TCRbulk_Spleen_DNA_100p0ng_Replicate-2_2021-11-15_S65_L001_R1_001.fa...`, `HPAP-005_TCRbulk_Spleen_DNA_100ng_Replicate-1_2021-10-22_S66_L001_R2_001.fast...`, `HPAP-005_TCRbulk_Spleen_DNA_100ng_Replicate-2_2021-10-22_S67_L001_R1_001.fast...`, `HPAP-005_TCRbulk_Spleen_DNA_100p0ng_Replicate-1_2021-11-15_S66_L001_R1_001.fa...`, `HPAP-005_TCRbulk_Spleen_DNA_100p0ng_Replicate-2_2021-11-15_S67_L001_R1_001.fa...`, `HPAP-006_TCRbulk_Spleen_DNA_100ng_Replicate-1_2021-10-22_S68_L001_R1_001.fast...`, `HPAP-006_TCRbulk_Spleen_DNA_100ng_Replicate-2_2021-10-22_S69_L001_R1_001.fast...`, `HPAP-006_TCRbulk_Spleen_DNA_100p0ng_Replicate-1_2021-11-15_S68_L001_R1_001.fa...`, `HPAP-006_TCRbulk_Spleen_DNA_100p0ng_Replicate-2_2021-11-15_S69_L001_R1_001.fa...`, `HPAP-007_TCRbulk_Spleen_DNA_100ng_Replicate-1_2021-10-22_S3_L001_R2_001.fastq...`, `HPAP-007_TCRbulk_Spleen_DNA_100ng_Replicate-2_2021-10-22_S4_L001_R1_001.fastq...`, `HPAP-007_TCRbulk_Spleen_DNA_100p0ng_Replicate-1_2021-11-15_S3_L001_R1_001.fas...`, `HPAP-007_TCRbulk_Spleen_DNA_100p0ng_Replicate-2_2021-11-15_S4_L001_R1_001.fas...`, `HPAP-008_TCRbulk_Spleen_DNA_100ng_Replicate-1_2021-10-22_S70_L001_R1_001.fast...`, `HPAP-008_TCRbulk_Spleen_DNA_100ng_Replicate-2_2021-10-22_S71_L001_R2_001.fast...`, `HPAP-008_TCRbulk_Spleen_DNA_100p0ng_Replicate-1_2021-11-15_S70_L001_R1_001.fa...`, `HPAP-008_TCRbulk_Spleen_DNA_100p0ng_Replicate-2_2021-11-15_S71_L001_R1_001.fa...`, `HPAP-009_TCRbulk_Spleen_DNA_100ng_Replicate-1_2021-10-22_S72_L001_R1_001.fast...`, `HPAP-009_TCRbulk_Spleen_DNA_100ng_Replicate-2_2021-10-22_S73_L001_R2_001.fast...` ... (>30 distinct values)
- **`Source`** (1 distinct): `Luning Prak Lab`
- **`Contact`** (1 distinct): `Adil Mohammed`

### `scATAC-seq` (47 rows)

```
CREATE TABLE `scATAC-seq` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-035`, `HPAP-036`, `HPAP-039`, `HPAP-040`, `HPAP-042`, `HPAP-044`, `HPAP-045`, `HPAP-047`, `HPAP-049`, `HPAP-050`, `HPAP-051`, `HPAP-052`, `HPAP-053`, `HPAP-054`, `HPAP-055`, `HPAP-056`, `HPAP-057`, `HPAP-058`, `HPAP-059`, `HPAP-061`, `HPAP-062`, `HPAP-063`, `HPAP-064`, `HPAP-066`, `HPAP-067`, `HPAP-069`, `HPAP-072`, `HPAP-075`, `HPAP-077`, `HPAP-079` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `scATAC-seq`
- **`Tissue`** (1 distinct): `Islet`
- **`Source`** (1 distinct): `Klaus Kaestner_Upenn`
- **`Contact`** (1 distinct): `PanKbase/Kai Liu`

### `scRNA-seq` (79 rows)

```
CREATE TABLE `scRNA-seq` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Cell_Type` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Source` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-019`, `HPAP-020`, `HPAP-021`, `HPAP-022`, `HPAP-023`, `HPAP-024`, `HPAP-026`, `HPAP-028`, `HPAP-029`, `HPAP-032`, `HPAP-034`, `HPAP-035`, `HPAP-036`, `HPAP-037`, `HPAP-038`, `HPAP-039`, `HPAP-040`, `HPAP-042`, `HPAP-043`, `HPAP-044`, `HPAP-045`, `HPAP-047`, `HPAP-049`, `HPAP-050`, `HPAP-051`, `HPAP-052`, `HPAP-053`, `HPAP-054`, `HPAP-055`, `HPAP-056` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `scRNA-seq`
- **`Cell_Type`** (2 distinct): `beta`, `Islet cells`
- **`Tissue`** (1 distinct): `Islet`
- **`Source`** (2 distinct): ``, `Kaestner Lab_Upenn`
- **`Contact`** (1 distinct): `PanKbase/Kai Liu`

### `snMultiomics` (106 rows)

```
CREATE TABLE `snMultiomics` (
  `Donor` VARCHAR(50),
  `Data Modality` VARCHAR(255),
  `Tissue` VARCHAR(255),
  `Region` VARCHAR(255),
  `Contact` VARCHAR(255)
);
```

**Column values:**

- **`Donor`** (31 distinct): `HPAP-008`, `HPAP-012`, `HPAP-019`, `HPAP-020`, `HPAP-023`, `HPAP-024`, `HPAP-028`, `HPAP-029`, `HPAP-030`, `HPAP-032`, `HPAP-038`, `HPAP-041`, `HPAP-045`, `HPAP-071`, `HPAP-079`, `HPAP-084`, `HPAP-087`, `HPAP-093`, `HPAP-095`, `HPAP-096`, `HPAP-097`, `HPAP-102`, `HPAP-104`, `HPAP-107`, `HPAP-112`, `HPAP-114`, `HPAP-116`, `HPAP-117`, `HPAP-119`, `HPAP-122` ... (>30 distinct values)
- **`Data Modality`** (1 distinct): `snMultiomics`
- **`Tissue`** (4 distinct): `Lymph_node`, `Pancreas`, `PBMC`, `Spleen`
- **`Region`** (4 distinct): ``, `Body`, `Head`, `Tail`
- **`Contact`** (1 distinct): `Haoxuan Zeng`

