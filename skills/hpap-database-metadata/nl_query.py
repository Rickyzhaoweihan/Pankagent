"""
Natural language query interface for the HPAP T1D database.

Usage:
    python nl_query.py "how many donors have T1D?"
    python nl_query.py "show me GADA positive donors"
    python nl_query.py   # interactive mode
"""

import sys
import os
import json

import anthropic
import pymysql
import pymysql.cursors

SCHEMA = open(os.path.join(os.path.dirname(__file__), "schema.md")).read()

DB_CONFIG = {
    "host": "mysql-database-nlb-3f645cf8839d37de.elb.us-east-1.amazonaws.com",
    "user": "alb_user",
    "password": "Lamia14-5",
    "port": 3306,
    "charset": "utf8mb4",
    "connect_timeout": 10,
    "read_timeout": 30,
}

SYSTEM_PROMPT = f"""You are an expert MySQL query writer for the HPAP (Human Pancreas Analysis Program) Type 1 Diabetes research database.

There are two databases on the same MySQL server: `donors` and `modalities`.

Here is the full schema:

{SCHEMA}

IMPORTANT — actual values in the `clinical_diagnosis` / `Clinical Diagnosis` columns:

In donors.Metadata (`clinical_diagnosis`):
  'T1DM', 'T1DM Recent', 'T1DM/MODY', 'T2DM', 'T2DM Gastric Bypass', 'ND', '???'

In donors.AAb_cPeptide_Metadata and donors.Summary_by_Diagnosis (`Clinical Diagnosis`):
  'T1DM', 'T1DM (recent DKA)', 'T1DM Recent onset', 'Recent T1DM Unsuspected',
  'T1DM or MODY, undetermined', 'T1D control', 'T2DM', 'T2DM Gastric bypass',
  'T2DM polycystic ovaries', 'T2DM (? prediabetic)', 'T2D control'

When the user says "T1D" they mean all T1DM variants — use LIKE '%T1D%' or list them explicitly.
When the user says "T2D" they mean all T2DM variants.
"ND" means non-diabetic / healthy control.

RULES:
- Always wrap table and column names in backticks — many contain spaces, parentheses, and special characters.
- The donor ID column is named differently across tables:
  - donors.Metadata: `donor_ID` (also has `Donor`)
  - donors.AAb_cPeptide_Metadata: `Donor ID`
  - donors.cell_counts: `Donor ID`
  - All modalities tables: `Donor`
- For cross-database queries, prefix tables: donors.`Metadata`, modalities.`scRNA-seq`
- Some numeric-looking columns in Metadata are VARCHAR (age_years, BMI, etc.) — cast if doing math.
- All columns are nullable.
- Always add LIMIT for exploratory queries.
- Write MySQL-compatible SQL only.

Respond with ONLY a JSON object (no markdown, no explanation) with these keys:
- "database": which database to run against ("donors" or "modalities")
- "sql": the MySQL query string
- "explanation": one-sentence explanation of what the query does
"""

def nl_to_sql(question: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def run_query(database: str, sql: str) -> list[dict]:
    conn = pymysql.connect(**DB_CONFIG, database=database)
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def format_results(rows: list[dict]) -> str:
    if not rows:
        return "(no results)"

    cols = list(rows[0].keys())
    widths = [max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]

    header = " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
    return "\n".join(lines)


def ask(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 60)

    result = nl_to_sql(question)
    database = result["database"]
    sql = result["sql"]
    explanation = result.get("explanation", "")

    print(f"Database: {database}")
    print(f"SQL: {sql}")
    if explanation:
        print(f"Explanation: {explanation}")
    print("-" * 60)

    rows = run_query(database, sql)
    print(f"Results ({len(rows)} rows):\n")
    print(format_results(rows))


def main():
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("HPAP Natural Language Query (type 'quit' to exit)")
        print("=" * 60)
        while True:
            try:
                question = input("\nAsk: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in ("quit", "exit", "q"):
                break
            try:
                ask(question)
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
