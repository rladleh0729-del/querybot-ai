"""Queryable answer ratings stored in the existing D1 database."""
import hashlib
import threading
import time

import acct_store

_lock = threading.Lock()
_initialized = None
TABLE = """CREATE TABLE IF NOT EXISTS answer_feedback (
 id TEXT PRIMARY KEY, rater TEXT NOT NULL, rating INTEGER NOT NULL CHECK(rating IN (-1,1)),
 question TEXT NOT NULL, answer TEXT NOT NULL, model TEXT NOT NULL,
 created INTEGER NOT NULL, updated INTEGER NOT NULL)"""


def _query(sql, params=None):
    if not acct_store.ready():
        raise acct_store.StoreError("not ready")
    if acct_store._state["kind"] == "sqlite":
        with acct_store._lock:
            cur = acct_store._sqlite.execute(sql, params or [])
            rows = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()] if cur.description else []
            acct_store._sqlite.commit()
            return rows
    return acct_store._d1(sql, params)


def setup():
    global _initialized
    with _lock:
        identity = (acct_store._state["db"], id(acct_store._sqlite))
        if _initialized != identity:
            _query(TABLE)
            _query("CREATE INDEX IF NOT EXISTS answer_feedback_updated ON answer_feedback(updated DESC)")
            _initialized = identity


def save(rater, rating, question, answer, model):
    setup()
    rid = hashlib.sha256((rater + "\0" + question + "\0" + answer + "\0" + model).encode()).hexdigest()
    now = int(time.time() * 1000)
    if rating == 0:
        _query("DELETE FROM answer_feedback WHERE id=? AND rater=?", [rid, rater])
    else:
        _query("""INSERT INTO answer_feedback(id,rater,rating,question,answer,model,created,updated)
         VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET rating=excluded.rating, updated=excluded.updated""",
               [rid, rater, rating, question, answer, model, now, now])
    return now


def list_recent(rating=0, limit=100):
    setup()
    where = " WHERE rating=?" if rating in (-1, 1) else ""
    params = [rating] if where else []
    rows = _query("SELECT rating,question,answer,model,created,updated FROM answer_feedback" + where +
                  " ORDER BY updated DESC,id DESC LIMIT ?", params + [min(100, max(1, limit))])
    counts = _query("SELECT COUNT(*) AS total, COALESCE(SUM(rating=1),0) AS likes, COALESCE(SUM(rating=-1),0) AS dislikes FROM answer_feedback")[0]
    return {"items": rows, "counts": counts}
