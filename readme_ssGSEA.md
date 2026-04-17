# ssGSEA Server

HTTP server for running ssGSEA (single-sample Gene Set Enrichment Analysis) on pseudo-bulk immune cell data. Loads a Seurat scRNA-seq object at startup, filters to immune cells, aggregates to donor-level pseudo-bulk profiles, and exposes ssGSEA scoring via a REST API.

## Quick Start

```bash
cd /home/ec2-user/r-home
Rscript server_ssGSEA.R
```

Startup takes a while (loading and processing the Seurat object). Once you see:

```
ssGSEA server running on http://Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com
```

The server is ready to accept requests.

### Run in background

```bash
nohup Rscript /home/ec2-user/r-home/server_ssGSEA.R > ssgsea.log 2>&1 &
```

Check logs:

```bash
tail -f ssgsea.log
```

## Endpoints

### GET /help

API documentation with endpoint descriptions and example usage.

```bash
curl -s http://Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com/help | python3 -m json.tool
```

### GET /genes

Returns all available gene names in the dataset.

```bash
curl -s http://Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com/genes | python3 -m json.tool
```

Search by prefix:

```bash
curl -s "http://Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com/genes?search=INS" | python3 -m json.tool
```

Example response:

```json
{
    "search": "INS",
    "count": 16,
    "genes": ["INS", "INS-IGF2", "INSC", "INSIG1", "INSIG2", ...]
}
```

### GET /donors

Returns donor metadata (one row per donor).

```bash
curl -s http://Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com/donors | python3 -m json.tool
```

### POST /ssgsea

Run ssGSEA with a custom gene set. Send a JSON body with a `genes` array.

```bash
curl -s -X POST http://Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com/ssgsea \
  -H "Content-Type: application/json" \
  -d '{"genes":["INS","GCG","SST","PPY"]}' \
  | python3 -m json.tool
```

Example response:

```json
{
    "genes_submitted": 4,
    "genes_used": 4,
    "genes_not_found": [],
    "scores": [
        { "donor_id": "donor_A", "score": 0.42 },
        { "donor_id": "donor_B", "score": 0.31 }
    ]
}
```

If some genes aren't found in the dataset, they'll appear in `genes_not_found` and the analysis will proceed with the valid ones.

## Data

- **Seurat object:** `/home/ec2-user/r-data/060425_scRNA_v3.3.rds`
- **Cell type filter:** `Immune`
- **Donor column:** `center_donor_id`
- **Normalization:** logCPM (log2 of counts per million + 1)

## Notes

- The server binds to `Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com`.
- All heavy computation (loading, filtering, pseudo-bulking, normalizing) happens once at startup. Each `/ssgsea` request only runs the ssGSEA scoring step.
- To stop the server: `kill $(lsof -t -i:9030)`
