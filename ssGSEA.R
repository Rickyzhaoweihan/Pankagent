############################################################
# Quick ssGSEA script for scRNA-seq data
#
# Goal
# ----
# 1. Load the Seurat object
# 2. Keep only cells with Cell_Type == "Immune"
# 3. Aggregate counts across all donors at the donor level
#    (one pseudo-bulk expression profile per donor)
# 4. Convert to CPM and logCPM
# 5. Build one RANDOM gene set as example input
# 6. Run ssGSEA across donors
# 7. Export results to CSV
#
# Notes
# -----
# - This script is written to be easy to read, even if you are
#   not very familiar with bioinformatics.
# - "Pseudo-bulk" means we sum raw counts across cells that
#   belong to the same donor, then normalize.
# - ssGSEA gives one enrichment score per gene set per sample.
#   Here, each "sample" is one donor-level pseudo-bulk profile.
############################################################

############################
# 0) Load packages
############################
# Install these if needed:
# install.packages(c("Matrix", "dplyr", "tibble"))
# if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
# BiocManager::install("GSVA")

library(Matrix)
library(dplyr)
library(tibble)
library(GSVA)

############################
# 1) File paths
############################
rds_path <- "/home/ec2-user/r-data/060425_scRNA_v3.3.rds"
out_dir  <- "/home/ec2-user/r-data/psuedo expression result"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

############################
# 2) Load the Seurat object
############################
obj <- readRDS(rds_path)

############################
# 3) Extract metadata and counts
############################
# Metadata: one row per cell
meta <- obj@meta.data %>%
  rownames_to_column("cell_id")

# Raw counts matrix from the RNA assay
# Rows = genes
# Columns = cells
counts <- LayerData(obj, assay = "RNA", layer = "counts")

# Make sure metadata rows are aligned with count matrix columns
meta <- meta[match(colnames(counts), meta$cell_id), , drop = FALSE]

# Safety check: every metadata row should now match the correct cell
stopifnot(all(meta$cell_id == colnames(counts)))

############################
# 4) Inspect available Cell_Type labels
############################
print(sort(unique(meta$Cell_Type)))

############################
# 5) Keep only immune cells
############################
# IMPORTANT:
# This script uses exact matching to "Immune".
# If your label is actually "Immune(Macrophages)" or something similar,
# either change the line below or inspect unique(meta$Cell_Type) first.
immune_label <- "Immune"

keep_immune <- !is.na(meta$Cell_Type) & meta$Cell_Type == immune_label

meta_immune   <- meta[keep_immune, , drop = FALSE]
counts_immune <- counts[, keep_immune, drop = FALSE]

cat("Number of immune cells kept:", ncol(counts_immune), "\n")

if (ncol(counts_immune) == 0) {
  stop("No cells found with Cell_Type == 'Immune'. Check the exact label in meta$Cell_Type.")
}

############################
# 6) Keep only cells with donor ID
############################
donor_col <- "center_donor_id"

if (!donor_col %in% colnames(meta_immune)) {
  stop(paste("Metadata column not found:", donor_col))
}

keep_donor <- !is.na(meta_immune[[donor_col]]) & meta_immune[[donor_col]] != ""

meta_immune   <- meta_immune[keep_donor, , drop = FALSE]
counts_immune <- counts_immune[, keep_donor, drop = FALSE]

cat("Number of immune cells with donor ID:", ncol(counts_immune), "\n")

if (ncol(counts_immune) == 0) {
  stop("No immune cells remain after filtering for non-missing donor ID.")
}

############################
# 7) Build donor-level pseudo-bulk counts
############################
# We want one expression profile per donor.
# To do this:
# - all immune cells from donor A are summed together
# - all immune cells from donor B are summed together
# - etc.

# Create donor factor
donor_factor <- factor(meta_immune[[donor_col]])

# Build a sparse design matrix:
# rows = cells
# columns = donors
# A cell gets a 1 in the column for its donor
design_mat <- sparse.model.matrix(~ 0 + donor_factor)
colnames(design_mat) <- levels(donor_factor)

# Pseudo-bulk counts:
# genes x cells  multiplied by  cells x donors  = genes x donors
pb_counts <- counts_immune %*% design_mat

cat("Pseudo-bulk count matrix dimensions:\n")
print(dim(pb_counts))  # genes x donors

############################
# 8) Normalize to CPM and logCPM
############################
# CPM = counts per million
# This adjusts for different donor library sizes

library_size <- Matrix::colSums(pb_counts)

# Avoid division by zero just in case
if (any(library_size == 0)) {
  stop("At least one donor has zero total counts after pseudo-bulk aggregation.")
}

pb_cpm <- t(t(pb_counts) / library_size) * 1e6

# logCPM is commonly used because raw CPM can be very skewed
pb_logcpm <- log2(pb_cpm + 1)

############################
# 9) Create a random gene list as example input
############################
# This is ONLY an example input gene set.
# In real analysis, replace this with your real pathway / marker gene list.

set.seed(123)

all_genes <- rownames(pb_logcpm)

# Example: choose 50 random genes
# If there are fewer than 50 genes, use all available genes
n_random_genes <- min(50, length(all_genes))

random_gene_set <- sample(all_genes, size = n_random_genes, replace = FALSE)

# GSVA/ssGSEA expects a named list of gene sets
gene_sets <- list(Random_Gene_Set = random_gene_set)

cat("Random gene set size:", length(random_gene_set), "\n")

############################
# 10) Run ssGSEA
############################
# Input matrix for ssGSEA:
# rows = genes
# columns = donors
#
# We use logCPM as the expression matrix.
# ssGSEA ranks genes within each donor/sample and computes an
# enrichment score for the chosen gene set.

expr_mat <- as.matrix(pb_logcpm)

# Different GSVA versions use slightly different interfaces.
# This block tries the newer interface first, then falls back
# to the older one if needed.

ssgsea_scores <- NULL

try_new <- try({
  param <- ssgseaParam(exprData = expr_mat, geneSets = gene_sets)
  ssgsea_scores <- gsva(param)
}, silent = TRUE)

if (is.null(ssgsea_scores)) {
  try_old <- try({
    ssgsea_scores <- gsva(
      expr = expr_mat,
      gset.idx.list = gene_sets,
      method = "ssgsea",
      kcdf = "Gaussian",
      abs.ranking = FALSE,
      verbose = TRUE
    )
  }, silent = TRUE)
}

if (is.null(ssgsea_scores)) {
  stop("ssGSEA failed. Check that the GSVA package is installed and compatible with your R version.")
}

cat("ssGSEA score matrix dimensions:\n")
print(dim(ssgsea_scores))  # gene sets x donors

############################
# 11) Make outputs easy to read
############################

# A) Donor-level metadata summary
# Here we keep one row per donor from the immune-cell metadata.
# Only columns that are constant within donor are ideal donor metadata,
# but for a quick summary we keep distinct donor IDs and a few useful fields if present.

candidate_meta_cols <- c(
  donor_col,
  "source",
  "sex",
  "age_(years)",
  "aliases",
  "bmi",
  "diabetes_status",
  "description_of_diabetes_status",
  "hba1c_(percentage)",
  "treatments",
  "chemistry"
)

candidate_meta_cols <- candidate_meta_cols[candidate_meta_cols %in% colnames(meta_immune)]

donor_meta <- meta_immune %>%
  select(any_of(candidate_meta_cols)) %>%
  distinct(.data[[donor_col]], .keep_all = TRUE) %>%
  arrange(.data[[donor_col]])

# B) CPM table
cpm_df <- as.data.frame(as.matrix(pb_cpm)) %>%
  rownames_to_column("gene")

# C) logCPM table
logcpm_df <- as.data.frame(as.matrix(pb_logcpm)) %>%
  rownames_to_column("gene")

# D) ssGSEA score table
# transpose so rows = donors, columns = gene sets
ssgsea_df <- as.data.frame(t(as.matrix(ssgsea_scores))) %>%
  rownames_to_column("center_donor_id")

# E) Save the random gene list itself
random_gene_df <- data.frame(
  gene_set = "Random_Gene_Set",
  gene = random_gene_set,
  stringsAsFactors = FALSE
)

############################
# 12) Write CSV outputs
############################
write.csv(
  donor_meta,
  file.path(out_dir, "Immune_all_donors_donor_metadata_for_ssGSEA.csv"),
  row.names = FALSE
)

write.csv(
  cpm_df,
  file.path(out_dir, "Immune_all_donors_pseudobulk_CPM.csv"),
  row.names = FALSE
)

write.csv(
  logcpm_df,
  file.path(out_dir, "Immune_all_donors_pseudobulk_logCPM.csv"),
  row.names = FALSE
)

write.csv(
  ssgsea_df,
  file.path(out_dir, "Immune_all_donors_ssGSEA_scores_random_gene_set.csv"),
  row.names = FALSE
)

write.csv(
  random_gene_df,
  file.path(out_dir, "Immune_all_donors_random_gene_set_used_for_ssGSEA.csv"),
  row.names = FALSE
)

############################
# 13) Print a short summary
############################
cat("\nDone.\n")
cat("Output folder:\n", out_dir, "\n\n")

cat("Number of genes:", nrow(pb_logcpm), "\n")
cat("Number of immune donors:", ncol(pb_logcpm), "\n")
cat("Random gene set size:", length(random_gene_set), "\n\n")

cat("Files written:\n")
print(list.files(out_dir, pattern = "Immune_all_donors", full.names = TRUE))


############################################################
# OPTIONAL: How to replace the random gene set with your own
############################################################
# Example:
# my_gene_set <- c("INS", "GCG", "SST", "PPY")
# gene_sets <- list(My_Custom_Set = my_gene_set)
#
# Then rerun section 10 onward.
############################################################
