from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EnrollRecord:
    id: str
    person_id: str
    name: str | None
    phone: str | None
    model: str
    dim: int
    created_at: str  # ISO8601 UTC


@dataclass(frozen=True)
class StoredEmbedding:
    id: str
    person_id: str
    name: str | None
    model: str
    dim: int
    embedding: np.ndarray  # float32, shape (dim,)
    created_at: str


@dataclass(frozen=True)
class PersonSummary:
    person_id: str
    name: str
    phone: str | None
    embeddings: int
    last_embedding_at: str | None
    created_at: str
    updated_at: str


class EmbeddingStore:
    def __init__(self, db_path: str | os.PathLike):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                  person_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  phone TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "people", "phone", "TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                  id TEXT PRIMARY KEY,
                  person_id TEXT NOT NULL,
                  model TEXT NOT NULL,
                  dim INTEGER NOT NULL,
                  embedding BLOB NOT NULL,
                  det_score REAL,
                  bbox_x1 REAL,
                  bbox_y1 REAL,
                  bbox_x2 REAL,
                  bbox_y2 REAL,
                  created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_person_id ON embeddings(person_id);"
            )

    @staticmethod
    def _to_blob(embedding: np.ndarray) -> bytes:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return emb.tobytes(order="C")

    @staticmethod
    def _from_blob(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != dim:
            # 容错：若 dim 记录不一致，仍尽量返回实际长度
            return arr.astype(np.float32).reshape(-1)
        return arr.astype(np.float32).reshape(dim)

    def enroll(
        self,
        *,
        person_id: str,
        name: str | None,
        phone: str,
        model: str,
        embedding: np.ndarray,
        det_score: float | None = None,
        bbox_xyxy: tuple[float, float, float, float] | None = None,
    ) -> EnrollRecord:
        rid = str(uuid.uuid4())
        emb = self._to_blob(embedding)
        dim = int(np.asarray(embedding, dtype=np.float32).reshape(-1).shape[0])
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        x1 = y1 = x2 = y2 = None
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = bbox_xyxy

        with self._connect() as conn:
            if name is not None and name.strip():
                conn.execute(
                    """
                    INSERT INTO people (person_id, name, phone, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(person_id) DO UPDATE SET
                      name = excluded.name,
                      phone = excluded.phone,
                      updated_at = excluded.updated_at;
                    """,
                    (person_id, name.strip(), phone.strip(), now, now),
                )
            conn.execute(
                """
                INSERT INTO embeddings (
                  id, person_id, model, dim, embedding,
                  det_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                  created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (rid, person_id, model, dim, emb, det_score, x1, y1, x2, y2, now),
            )

        return EnrollRecord(
            id=rid,
            person_id=person_id,
            name=name.strip() if name else None,
            phone=phone.strip(),
            model=model,
            dim=dim,
            created_at=now,
        )

    def count_person(self, person_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS c FROM embeddings WHERE person_id = ?;",
                (person_id,),
            ).fetchone()
            return int(row["c"] if row else 0)

    def get_person_name(self, person_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM people WHERE person_id = ?;",
                (person_id,),
            ).fetchone()
            return str(row["name"]) if row else None

    def list_people(self) -> list[PersonSummary]:
        sql = """
        SELECT
          p.person_id,
          p.name,
          p.phone,
          p.created_at,
          p.updated_at,
          COUNT(e.id) AS embeddings,
          MAX(e.created_at) AS last_embedding_at
        FROM people p
        LEFT JOIN embeddings e ON e.person_id = p.person_id
        GROUP BY p.person_id
        ORDER BY p.updated_at DESC;
        """
        out: list[PersonSummary] = []
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
            for r in rows:
                out.append(
                    PersonSummary(
                        person_id=str(r["person_id"]),
                        name=str(r["name"]),
                        phone=str(r["phone"]) if r["phone"] is not None else None,
                        embeddings=int(r["embeddings"] or 0),
                        last_embedding_at=str(r["last_embedding_at"]) if r["last_embedding_at"] is not None else None,
                        created_at=str(r["created_at"]),
                        updated_at=str(r["updated_at"]),
                    )
                )
        return out

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, definition: str):
        cols = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
        exists = any(str(c["name"]) == column_name for c in cols)
        if not exists:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition};")

    def delete_person(self, person_id: str) -> dict:
        """
        删除一个人及其所有 embedding。
        返回删除计数，便于接口响应。
        """
        with self._connect() as conn:
            # 先统计
            row = conn.execute(
                "SELECT COUNT(1) AS c FROM embeddings WHERE person_id = ?;",
                (person_id,),
            ).fetchone()
            emb_count = int(row["c"] if row else 0)

            # 事务性删除
            conn.execute("DELETE FROM embeddings WHERE person_id = ?;", (person_id,))
            cur = conn.execute("DELETE FROM people WHERE person_id = ?;", (person_id,))
            people_deleted = int(cur.rowcount or 0)

        return {"person_id": person_id, "people_deleted": people_deleted, "embeddings_deleted": emb_count}

    def list_all_embeddings(self, model: str | None = None) -> list[StoredEmbedding]:
        sql = """
        SELECT
          e.id, e.person_id, p.name, e.model, e.dim, e.embedding, e.created_at
        FROM embeddings e
        LEFT JOIN people p ON p.person_id = e.person_id
        """
        args: tuple[object, ...] = ()
        if model is not None:
            sql += " WHERE e.model = ?"
            args = (model,)
        sql += " ORDER BY e.created_at ASC"

        out: list[StoredEmbedding] = []
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
            for r in rows:
                out.append(
                    StoredEmbedding(
                        id=str(r["id"]),
                        person_id=str(r["person_id"]),
                        name=str(r["name"]) if r["name"] is not None else None,
                        model=str(r["model"]),
                        dim=int(r["dim"]),
                        embedding=self._from_blob(r["embedding"], int(r["dim"])),
                        created_at=str(r["created_at"]),
                    )
                )
        return out

