"""Tiny SQLite store for past analyses."""
import json
import sqlite3
from datetime import datetime

from .config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, sport TEXT, filename TEXT,
        action TEXT, confidence REAL, overall REAL, scores TEXT)""")
    return c


def save_result(sport, filename, result):
    with _conn() as c:
        c.execute("INSERT INTO analyses(created,sport,filename,action,confidence,overall,scores) VALUES(?,?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), sport, filename, result["action"],
                   result["confidence"], result["overall"], json.dumps(result["scores"])))


def load_history(limit=50):
    with _conn() as c:
        rows = c.execute("SELECT created,sport,filename,action,confidence,overall,scores FROM analyses "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(created=r[0], sport=r[1], filename=r[2], action=r[3], confidence=r[4],
                 overall=r[5], scores=json.loads(r[6])) for r in rows]
