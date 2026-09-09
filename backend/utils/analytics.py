"""
Description: Analytics and logging utility. Logs user queries, answers,
    and metadata to a simple SQLite database. Helps track usage, cache hits,
    and response times, addressing the PRD's requirement for basic analytics.
Inputs: None (defines QueryLogger class).
Outputs: None.
Dependencies: sqlite3, json, time, datetime.
Utilities: used by asgi.py and cli.py.
"""

import sqlite3
import json
import time
from datetime import datetime
import os


class QueryLogger:
    """
    Description: Thread-safe logger that writes query data to an SQLite file.
    Inputs (constructor): db_path (path to the sqlite database file).
    Utilities: used by asgi.py and cli.py to log interactions and fetch stats.
    """

    def __init__(self, db_path="analytics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the analytics table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    question TEXT,
                    answer TEXT,
                    source_urls TEXT,
                    response_time_ms REAL,
                    cache_hit BOOLEAN,
                    platform TEXT
                )
            ''')
            conn.commit()

    def log_query(self, question, answer, source_urls=None, response_time_ms=0, cache_hit=False, platform="unknown"):
        """
        Description: Logs a single query interaction.
        Inputs: question, answer, source_urls (list), response_time_ms (float),
            cache_hit (bool), platform (string: e.g., 'whatsapp' or 'cli' or 'chat').
        Outputs: none.
        Dependencies: sqlite3.
        """
        if source_urls is None:
            source_urls = []
        
        timestamp = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO queries 
                (timestamp, question, answer, source_urls, response_time_ms, cache_hit, platform)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                question,
                answer,
                json.dumps(source_urls),
                response_time_ms,
                cache_hit,
                platform
            ))
            conn.commit()

    def get_recent_queries(self, limit=50):
        """
        Description: Retrieves the most recent queries.
        Inputs: limit (int).
        Outputs: list of dicts.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            
        return [dict(row) for row in rows]

    def get_stats(self):
        """
        Description: Aggregates basic statistics from the logs.
        Inputs: None.
        Outputs: dict of stats.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM queries')
            total_queries = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM queries WHERE cache_hit = 1')
            total_cache_hits = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(response_time_ms) FROM queries')
            avg_response_time = cursor.fetchone()[0] or 0.0
            
        return {
            "total_queries": total_queries,
            "total_cache_hits": total_cache_hits,
            "hit_rate": round(total_cache_hits / total_queries, 3) if total_queries > 0 else 0.0,
            "avg_response_time_ms": round(avg_response_time, 2)
        }

# Shared singleton for the app
_query_logger = QueryLogger()

def get_query_logger():
    return _query_logger
