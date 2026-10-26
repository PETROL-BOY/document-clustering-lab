import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Dict

class VectorCache:
    def __init__(self, db_path: str = "./data/embedding_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    chunk_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (chunk_id, model, task_type)
                )
            """)
            conn.commit()

    def get(self, chunk_id: str, model: str, task_type: str) -> Optional[np.ndarray]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT dim, vector FROM embeddings_cache WHERE chunk_id = ? AND model = ? AND task_type = ?",
                (chunk_id, model, task_type)
            )
            row = cursor.fetchone()
            if row:
                dim, blob = row
                vec = np.frombuffer(blob, dtype=np.float32)
                return vec.copy()
        return None

    def set(self, chunk_id: str, model: str, task_type: str, vector: np.ndarray):
        vec_bytes = np.asarray(vector, dtype=np.float32).tobytes()
        dim = int(vector.shape[0])
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings_cache (chunk_id, model, task_type, dim, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, model, task_type, dim, vec_bytes)
            )
            conn.commit()

    def get_batch(self, chunk_ids: list[str], model: str, task_type: str) -> Dict[str, np.ndarray]:
        if not chunk_ids:
            return {}
        results = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(chunk_ids))
            query = f"""
                SELECT chunk_id, vector FROM embeddings_cache
                WHERE model = ? AND task_type = ? AND chunk_id IN ({placeholders})
            """
            cursor.execute(query, [model, task_type] + chunk_ids)
            for cid, blob in cursor.fetchall():
                results[cid] = np.frombuffer(blob, dtype=np.float32).copy()
        return results
