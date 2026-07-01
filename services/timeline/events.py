import sqlite3
import json
from datetime import datetime
import os

DB_PATH = os.path.join(os.getcwd(), "memory_events.db")

def init_event_db():
    """Initializes the events database if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event(event_type: str, description: str, metadata: dict = None) -> dict:
    """Logs a memory event to the database."""
    init_event_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    metadata_str = json.dumps(metadata) if metadata else "{}"
    
    cursor.execute(
        "INSERT INTO memory_events (timestamp, event_type, description, metadata) VALUES (?, ?, ?, ?)",
        (timestamp, event_type, description, metadata_str)
    )
    
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "id": event_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "description": description,
        "metadata": metadata or {}
    }

def get_events(limit: int = 100, offset: int = 0, event_type: str = None) -> list:
    """Retrieves list of logged events."""
    init_event_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if event_type:
        cursor.execute(
            "SELECT id, timestamp, event_type, description, metadata FROM memory_events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (event_type, limit, offset)
        )
    else:
        cursor.execute(
            "SELECT id, timestamp, event_type, description, metadata FROM memory_events ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        
    rows = cursor.fetchall()
    conn.close()
    
    events = []
    for r in rows:
        try:
            meta = json.loads(r[4])
        except Exception:
            meta = {}
        events.append({
            "id": r[0],
            "timestamp": r[1],
            "event_type": r[2],
            "description": r[3],
            "metadata": meta
        })
    return events

def get_events_before(timestamp: str) -> list:
    """Retrieves all events before a given ISO timestamp."""
    init_event_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, event_type, description, metadata FROM memory_events WHERE timestamp <= ? ORDER BY timestamp ASC",
        (timestamp,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    events = []
    for r in rows:
        try:
            meta = json.loads(r[4])
        except Exception:
            meta = {}
        events.append({
            "id": r[0],
            "timestamp": r[1],
            "event_type": r[2],
            "description": r[3],
            "metadata": meta
        })
    return events

# Initialize DB on load
init_event_db()
