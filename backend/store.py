"""
Batch result store: idempotency, conflict detection, async status and retention.

Backed by SQLite because the spec's requirements are exactly what a small
transactional store gives for free:

  idempotency  Re-sending the same batch_id returns the stored result instead of
               recomputing it -- which matters because the tester retries a batch
               on a retryable error, and a recompute would both cost money and
               quietly change the numbers being compared.
  409          The same batch_id carrying *different* questions is a tester-side
               mistake, not a retry, so it is rejected rather than silently
               overwriting a stored run.
  async        A batch that outgrows the sync deadline keeps being written here
               by its background thread, and GET reads its status.
  retention    Results are kept for at least 30 days, then purged.

A row is reserved before the work starts, inside one transaction, so two
concurrent sends of the same batch_id cannot both start computing.
"""
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend import config

STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

RESERVE_NEW = "new"
RESERVE_EXISTING = "existing"
RESERVE_CONFLICT = "conflict"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id      TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    request_hash  TEXT NOT NULL,
    status        TEXT NOT NULL,
    response_json TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS batches_run_id ON batches(run_id);
CREATE INDEX IF NOT EXISTS batches_expires_at ON batches(expires_at);
"""


def _connect() -> sqlite3.Connection:
    path: Path = config.STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # WAL so the background async worker can write while a GET reads.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def reserve(batch_id: str, run_id: str, request_hash: str) -> tuple[str, dict | None]:
    """Claim batch_id for this request.

    -> (RESERVE_NEW, None)            caller should compute
       (RESERVE_EXISTING, row)        stored result or in-flight batch
       (RESERVE_CONFLICT, row)        same id, different questions -> 409
    """
    created = _now()
    expires = created + timedelta(days=config.RESULT_RETENTION_DAYS)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if existing is not None:
            conn.execute("COMMIT")
            row = _row_to_dict(existing)
            if existing["request_hash"] != request_hash:
                return RESERVE_CONFLICT, row
            return RESERVE_EXISTING, row

        conn.execute(
            "INSERT INTO batches (batch_id, run_id, request_hash, status, response_json,"
            " created_at, expires_at) VALUES (?, ?, ?, ?, NULL, ?, ?)",
            (
                batch_id,
                run_id,
                request_hash,
                STATUS_PROCESSING,
                created.isoformat(timespec="seconds"),
                expires.isoformat(timespec="seconds"),
            ),
        )
        conn.execute("COMMIT")
    return RESERVE_NEW, None


def save(batch_id: str, status: str, response: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE batches SET status = ?, response_json = ? WHERE batch_id = ?",
            (status, json.dumps(response, ensure_ascii=False), batch_id),
        )


def get(batch_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
    return _row_to_dict(row) if row else None


def release(batch_id: str) -> None:
    """Drop a reservation whose work never ran, so the id can be retried.

    Without this, a crash between reserve() and save() would leave the batch
    permanently 'processing' and the tester unable to resend it.
    """
    with _connect() as conn:
        conn.execute(
            "DELETE FROM batches WHERE batch_id = ? AND response_json IS NULL", (batch_id,)
        )


def purge_expired() -> int:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM batches WHERE expires_at < ?", (_now().isoformat(),))
        return cur.rowcount or 0


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "batch_id": row["batch_id"],
        "run_id": row["run_id"],
        "request_hash": row["request_hash"],
        "status": row["status"],
        "response": json.loads(row["response_json"]) if row["response_json"] else None,
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
    }


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class RateLimiter:
    """Minimum interval between accepted batches, per API key.

    The spec sets a floor ("at least 1 batch/second must be accepted"), so this
    rejects only what arrives faster than that, and answers with the
    Retry-After the spec requires.
    """

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last: dict[str, float] = {}

    def check(self, key: str) -> float:
        """-> 0.0 if allowed, else seconds the caller should wait."""
        now = time.monotonic()
        last = self._last.get(key)
        if last is not None and (now - last) < self.min_interval_s:
            return self.min_interval_s - (now - last)
        self._last[key] = now
        return 0.0


rate_limiter = RateLimiter(config.RATE_LIMIT_MIN_INTERVAL_S)
