"""
HPAP Database Helper
====================

Utilities for connecting to and querying the HPAP T1D MySQL databases
(donors + modalities) hosted on AWS.

Usage:
    >>> from hpap_helper import HPAPDatabase
    >>> db = HPAPDatabase()
    >>> db.show_tables()
    >>> results = db.query("donors", "SELECT * FROM `Metadata` LIMIT 5")
    >>> db.preview("modalities", "scRNA-seq", n=3)
    >>> db.describe("donors", "Metadata")
"""

from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import time

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


class HPAPDatabase:
    """
    High-level interface to the HPAP T1D MySQL databases.

    Examples:
        >>> db = HPAPDatabase()
        >>> db.show_tables()
        >>> db.query("donors", "SELECT COUNT(*) FROM `Metadata`")
        >>> db.preview("modalities", "Flow_Cytometry", n=5)
        >>> db.describe("donors", "AAb_cPeptide_Metadata")
        >>> db.search_donors_by_diagnosis("T1D")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DB_CONFIG

    def connect(self, database: str) -> pymysql.Connection:
        if database not in DATABASES:
            raise ValueError(
                f"Unknown database '{database}'. Choose from: {DATABASES}"
            )
        return pymysql.connect(**self.config, database=database)

    @contextmanager
    def cursor(self, database: str):
        conn = self.connect(database)
        cur = conn.cursor(pymysql.cursors.DictCursor)
        try:
            yield cur
        finally:
            cur.close()
            conn.close()

    def query(
        self,
        database: str,
        sql: str,
        params: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as list of dicts.

        Args:
            database: "donors" or "modalities"
            sql: SQL query string
            params: Optional tuple of parameters for %s placeholders

        Returns:
            List of row dicts

        Example:
            >>> db.query("donors", "SELECT * FROM `Metadata` WHERE `sex` = %s LIMIT 5", ("M",))
        """
        with self.cursor(database) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def query_with_timing(
        self,
        database: str,
        sql: str,
        params: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Execute a query and return results with execution time.

        Returns:
            Dict with keys: rows, row_count, execution_time_s
        """
        start = time.time()
        rows = self.query(database, sql, params)
        elapsed = time.time() - start
        return {
            "rows": rows,
            "row_count": len(rows),
            "execution_time_s": round(elapsed, 3),
        }

    def show_tables(self, database: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List tables in one or both databases.

        Example:
            >>> db.show_tables()
            {'donors': ['AAb_cPeptide_Metadata', ...], 'modalities': ['BCR-seq', ...]}
        """
        dbs = [database] if database else DATABASES
        result = {}
        for db_name in dbs:
            with self.cursor(db_name) as cur:
                cur.execute("SHOW TABLES")
                result[db_name] = [list(r.values())[0] for r in cur.fetchall()]
        return result

    def describe(self, database: str, table: str) -> List[Dict[str, Any]]:
        """
        Return column info for a table (DESCRIBE output).

        Example:
            >>> cols = db.describe("donors", "Metadata")
            >>> for c in cols:
            ...     print(f"{c['Field']:40s} {c['Type']}")
        """
        with self.cursor(database) as cur:
            cur.execute(f"DESCRIBE `{table}`")
            return cur.fetchall()

    def row_count(self, database: str, table: str) -> int:
        """Return row count for a table."""
        with self.cursor(database) as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
            return cur.fetchone()["n"]

    def preview(
        self,
        database: str,
        table: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return first n rows from a table.

        Example:
            >>> rows = db.preview("modalities", "scRNA-seq", n=3)
        """
        with self.cursor(database) as cur:
            cur.execute(f"SELECT * FROM `{table}` LIMIT {int(n)}")
            return cur.fetchall()

    def search_donors_by_diagnosis(
        self, diagnosis: str
    ) -> List[Dict[str, Any]]:
        """
        Find donors matching a clinical diagnosis.

        Example:
            >>> db.search_donors_by_diagnosis("T1D")
        """
        return self.query(
            "donors",
            "SELECT `donor_ID`, `clinical_diagnosis`, `sex`, `age_years`, "
            "`Derived Diabetes Status` FROM `Metadata` "
            "WHERE `clinical_diagnosis` = %s",
            (diagnosis,),
        )

    def donor_modalities(self, donor_id: str) -> Dict[str, Any]:
        """
        Get modality availability flags for a donor from the Metadata table.

        Example:
            >>> db.donor_modalities("HPAP-001")
        """
        rows = self.query(
            "donors",
            "SELECT `donor_ID`, `scRNA-seq`, `scATAC-seq`, `snMultiomics`, "
            "`CITE-seq Protein`, `BCR-seq`, `TCR-seq`, `Bulk RNA-seq`, "
            "`Bulk ATAC-seq`, `Calcium Imaging`, `Flow Cytometry`, "
            "`Oxygen Consumption`, `Perifusion`, `CODEX`, `IMC`, `Histology` "
            "FROM `Metadata` WHERE `donor_ID` = %s",
            (donor_id,),
        )
        return rows[0] if rows else {}

    def inspect_all(self):
        """Print a full schema overview of both databases."""
        for db_name in DATABASES:
            tables = self.show_tables(db_name)[db_name]
            print(f"\n{'=' * 60}")
            print(f"  DATABASE: {db_name}")
            print(f"{'=' * 60}")
            print(f"\n  {len(tables)} table(s): {', '.join(tables)}\n")

            for table in tables:
                count = self.row_count(db_name, table)
                cols = self.describe(db_name, table)
                print(f"  +-- `{table}`  ({count} rows)")
                for col in cols:
                    nullable = "NULL" if col["Null"] == "YES" else "NOT NULL"
                    print(f"  |   {col['Field']:<45} {col['Type']:<20} {nullable}")
                print()


if __name__ == "__main__":
    db = HPAPDatabase()
    db.inspect_all()
