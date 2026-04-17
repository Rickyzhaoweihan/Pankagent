"""
Explore all tables in the HPAP databases and generate an enriched schema.md
that includes distinct values for each column.

Usage:
    python scripts/explore_schema.py > schema.md
"""

import pymysql
import pymysql.cursors

DB_CONFIG = {
    "host": "mysql-database-nlb-3f645cf8839d37de.elb.us-east-1.amazonaws.com",
    "user": "alb_user",
    "password": "Lamia14-5",
    "port": 3306,
    "charset": "utf8mb4",
    "connect_timeout": 10,
    "read_timeout": 30,
}

DATABASES = ["donors", "modalities"]

MAX_DISTINCT_DISPLAY = 30
MAX_VALUE_LENGTH = 80


def connect(database):
    return pymysql.connect(**DB_CONFIG, database=database)


def get_tables(cursor):
    cursor.execute("SHOW TABLES")
    return [list(r.values())[0] for r in cursor.fetchall()]


def get_columns(cursor, table):
    cursor.execute(f"DESCRIBE `{table}`")
    return cursor.fetchall()


def get_row_count(cursor, table):
    cursor.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
    return cursor.fetchone()["n"]


def get_distinct_values(cursor, table, column):
    cursor.execute(
        f"SELECT DISTINCT `{column}` AS v FROM `{table}` ORDER BY `{column}` LIMIT {MAX_DISTINCT_DISPLAY + 1}"
    )
    rows = cursor.fetchall()
    values = [r["v"] for r in rows]
    return values


def truncate(val, max_len=MAX_VALUE_LENGTH):
    s = str(val)
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


def explore_database(db_name):
    conn = connect(db_name)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    tables = get_tables(cursor)
    lines = []
    lines.append(f"## Database: `{db_name}`\n")
    lines.append(f"{len(tables)} tables.\n")

    for table in tables:
        row_count = get_row_count(cursor, table)
        columns = get_columns(cursor, table)
        lines.append(f"### `{table}` ({row_count} rows)\n")
        lines.append("```")
        lines.append(f"CREATE TABLE `{table}` (")

        col_defs = []
        for col in columns:
            col_defs.append(f"  `{col['Field']}` {col['Type'].upper()}")
        lines.append(",\n".join(col_defs))
        lines.append(");")
        lines.append("```\n")

        lines.append("**Column values:**\n")
        for col in columns:
            field = col["Field"]
            col_type = col["Type"].lower()

            distinct = get_distinct_values(cursor, table, field)
            n_distinct = len(distinct)
            truncated = n_distinct > MAX_DISTINCT_DISPLAY

            null_count_q = f"SELECT COUNT(*) AS n FROM `{table}` WHERE `{field}` IS NULL"
            cursor.execute(null_count_q)
            null_count = cursor.fetchone()["n"]

            non_null = [v for v in distinct if v is not None]

            if not non_null:
                lines.append(f"- **`{field}`**: all NULL")
                continue

            if truncated:
                shown = non_null[:MAX_DISTINCT_DISPLAY]
                formatted = [truncate(v) for v in shown]
                suffix = f" ... (>{MAX_DISTINCT_DISPLAY} distinct values)"
            else:
                formatted = [truncate(v) for v in non_null]
                suffix = ""

            null_note = f" + {null_count} NULL" if null_count > 0 else ""
            val_str = ", ".join(f"`{v}`" for v in formatted)
            lines.append(f"- **`{field}`** ({len(non_null)} distinct{null_note}): {val_str}{suffix}")

        lines.append("")

    cursor.close()
    conn.close()
    return "\n".join(lines)


def main():
    output = []
    output.append("# HPAP Database Schema Reference\n")
    output.append("Auto-generated with distinct values for every column.\n")

    for db_name in DATABASES:
        output.append(explore_database(db_name))

    print("\n".join(output))


if __name__ == "__main__":
    main()
