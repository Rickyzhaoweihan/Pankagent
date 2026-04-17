
# KNOWLEDGE GRAPH SCHEMA DOCUMENTATION
## Biomedical Research - Type 1 Diabetes Knowledge Graph

### Overview
This knowledge graph contains comprehensive biomedical data focused on Type 1 Diabetes (T1D) research. 
It integrates multiple data types including genomic, transcriptomic, epigenomic, and literature data.

**Total Files:** 19 (8 node files, 11 edge files)

### NODE TYPES


#### coding_elements;gene
**Description:** Genes from Ensembl database with genomic coordinates and annotations

**Properties:**
- `id`: String
- `id_version`: Int
- `description`: String
- `chr`: String
- `start_loc`: Int
- `end_loc`: Int
- `strand`: String
- `name`: String
- `GC_percentage`: Float
- `type`: String
- `link`: String
- `gencode_annotation`: String
- `data_version`: String
- `data_source`: String
- `tss`: Int
- `trans_length`: Int
- `count`: Int

**Source Files:**
- `Ensemble/opencypher/Ensembl_genes.node.csv` (100 samples)


#### unknown
**Description:** Node type: unknown

**Properties:**
- `data_version`: String
- `data_source`: String

**Source Files:**
- `GeneActivityScore/opencypher/scATACseq_OCR_supernode.edge.csv` (100 samples)


#### OCR
**Description:** Node type: OCR

**Properties:**
- `data_version`: String
- `data_source`: String

**Source Files:**
- `GeneActivityScore/opencypher/scATACseq_OCR_supernode.node.csv` (100 samples)


#### gene_ontology;ontology
**Description:** Genes from Ensembl database with genomic coordinates and annotations

**Properties:**
- `id`: String
- `link`: String
- `data_version`: String

**Source Files:**
- `GO/opencypher/go.node.csv` (100 samples)


#### disease;ontology
**Description:** Diseases from MONDO ontology, particularly Type 1 Diabetes

**Properties:**
- `url`: String
- `id`: String
- `definition`: String
- `name`: String
- `synonyms`: String
- `data_version`: String
- `data_source`: String

**Source Files:**
- `effector_gene/opencypher/T1D.node.csv` (1 samples)


#### cell_type;ontology
**Description:** Cell types from Cell Ontology, particularly pancreatic cell types

**Properties:**
- `url`: String
- `id`: String
- `definition`: String
- `name`: String
- `synonyms`: String
- `data_version`: String
- `data_source`: String

**Source Files:**
- `Expression/open_cypher/cell_line.csv` (8 samples)


#### sequence_variant;snp;variants
**Description:** Single nucleotide polymorphisms (SNPs) from NCBI database

**Properties:**
- `id`: String
- `link`: String
- `data_version`: String
- `data_source`: String

**Source Files:**
- `QTL/opencypher/snp.node.csv` (100 samples)


### EDGE TYPES (RELATIONSHIPS)


#### regulation
**Description:** Relationship type: regulation

**Relationship:** gene → gene

**Properties:**
- `biogrid_interaction_id`: Int
- `Entrez_ID_A`: Int
- `Entrez_ID_B`: Int
- `biogrid_id_interactor_a`: Int
- `biogrid_id_interactor_b`: Int
- `official_symbol_interactor_a`: String
- `official_symbol_interactor_b`: String
- `experimental_system`: String
- `experimental_system_type`: String
- `author`: String
- `publication_source`: String
- `organism_id_interactor_a`: Int
- `organism_id_interactor_b`: Int
- `throughput`: String
- `score`: Float
- `modification`: String
- `qualifications`: String
- `data_source`: String

**Source Files:**
- `BioGrid/opencypher/BIOGRID_with_ENSG_slim.csv` (100 samples)


#### OCR_activity
**Description:** Open chromatin region activity scores in different cell types and disease states

**Relationship:** open_chromatin_region → cell_type

**Properties:**
- `OCR_GeneActivityScore_mean`: Float
- `OCR_GeneActivityScore_median`: Float
- `non_diabetic__OCR_GeneActivityScore_mean`: Float
- `non_diabetic__OCR_GeneActivityScore_median`: Float
- `AAB_pos__OCR_GeneActivityScore_mean`: Float
- `AAB_pos__OCR_GeneActivityScore_median`: Float
- `type_1_diabetes__OCR_GeneActivityScore_mean`: Float
- `type_1_diabetes__OCR_GeneActivityScore_median`: Float
- `type_2_diabetes__OCR_GeneActivityScore_mean`: Float
- `type_2_diabetes__OCR_GeneActivityScore_median`: Float
- `data_version`: String
- `data_source`: String

**Source Files:**
- `GeneActivityScore/opencypher/scATACseq_GeneActivityScore.edge.csv` (100 samples)


#### function_annotation
**Description:** Gene Ontology annotations linking genes to their molecular functions, biological processes, and cellular components

**Relationship:** gene → gene_ontology

**Properties:**
- `data_version`: String
- `data_source`: String

**Source Files:**
- `GO/opencypher/go.edge.csv` (100 samples)


#### effector_gene
**Description:** Genes that are causal or almost certainly causal for Type 1 Diabetes based on genetic evidence

**Relationship:** gene → disease

**Properties:**
- `ResearchMethod`: String
- `CodingVariantEvidence`: String
- `ModelSystemEvidence`: String
- `EpigenomeEvidence`: String
- `QtlEvidence`: String
- `ConfidenceLevel`: String
- `data_source_url`: String
- `data_version`: String
- `data_source`: String

**Source Files:**
- `effector_gene/opencypher/effector_gene.edge.csv` (100 samples)


#### Differential_Expression
**Description:** Differential gene expression relationships between genes and cell types in disease vs control conditions

**Relationship:** gene → cell_type

**Properties:**
- `UpOrDownRegulation`: String
- `Log2FoldChange`: Float
- `SE_of_Log2FoldChange`: Float
- `P_value`: Float
- `Adjusted_P_value`: Float
- `data_version`: String
- `data_source`: String

**Source Files:**
- `Expression/open_cypher/20250606_DEG.edge.csv` (100 samples)


#### expression_level
**Description:** Relationship type: expression_level

**Relationship:** gene → cell_type

**Properties:**
- `NonDiabetic__expression_mean`: Float
- `NonDiabetic__expression_min`: Float
- `NonDiabetic__expression_25_quantile`: Float
- `NonDiabetic__expression_median`: Float
- `NonDiabetic__expression_75_quantile`: Float
- `NonDiabetic__expression_max`: Float
- `Type1Diabetic__expression_mean`: Float
- `Type1Diabetic__expression_min`: Float
- `Type1Diabetic__expression_25_quantile`: Float
- `Type1Diabetic__expression_median`: Float
- `Type1Diabetic__expression_75_quantile`: Float
- `Type1Diabetic__expression_max`: Float
- `All__expression_mean`: Float
- `All__expression_min`: Float
- `All__expression_25_quantile`: Float
- `All__expression_median`: Float
- `All__expression_75_quantile`: Float
- `All__expression_max`: Float
- `data_version`: String
- `data_source`: String

**Source Files:**
- `Expression/open_cypher/20250606_scRNAseq_pseudobulk_RUVseq.csv` (100 samples)


#### fine_mapped_eQTL
**Description:** Quantitative trait loci associations between genetic variants and gene expression

**Relationship:** snp → gene

**Properties:**
- `tissue_name`: String
- `tissue_id`: String
- `credible_set`: String
- `gene_name`: String
- `credibleset`: String
- `pip`: Float
- `nominal_p`: Float
- `lbf`: Float
- `effect_allele`: String
- `other_allele`: String
- `slope`: Float
- `n_snp`: Int
- `purity`: Float
- `data_version`: String
- `data_source`: String

**Source Files:**
- `QTL/opencypher/1_sQTL-gtex-susie.edge.csv` (100 samples)
- `QTL/opencypher/1_eQTL-gtex-susie.edge.csv` (100 samples)
- `QTL/opencypher/1_eQTL-inspire-susie.edge.csv` (100 samples)
- `QTL/opencypher/1_exonQTL-inspire-susie.edge.csv` (100 samples)


### USAGE GUIDELINES FOR LLM AGENTS

1. **Node Queries**: Use node types to understand what entities exist in the knowledge graph
2. **Relationship Queries**: Use edge types to understand how entities are connected
3. **Property Queries**: Use properties to filter and analyze specific attributes
4. **Multi-modal Analysis**: Combine different data types (genomic, expression, literature) for comprehensive analysis

### KEY DATA SOURCES
- **Ensembl**: Gene annotations and genomic coordinates
- **MONDO**: Disease ontologies, particularly Type 1 Diabetes
- **Gene Ontology**: Functional annotations
- **NCBI**: Genetic variants (SNPs)
- **BioGrid**: Protein-protein interactions
- **GTEx/INSPIRE**: Expression quantitative trait loci (eQTLs)
- **PanKbase**: Single-cell expression and chromatin accessibility data
- **HIRN**: Scientific literature and research articles

### COMMON QUERY PATTERNS
1. Find genes associated with Type 1 Diabetes
2. Identify differential expression patterns in pancreatic cell types
3. Map genetic variants to gene expression changes
4. Analyze protein-protein interaction networks
5. Explore functional annotations of disease-associated genes
