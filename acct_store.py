# 계정 저장소 (2026-11-15 사용자: "무료 서버에서 사용자의 대화 기록과 로그인 정보를 기억")
#  - 사용자 한 명 = 한 줄: uid(Google 계정 번호를 바꾼 값) · 대화 묶음(gzip → base64) · 저장 시각
#  - 진짜 저장소: Cloudflare D1(무료 5GB). Render 무료 서버는 껐다 켜면 파일이 사라져서 못 씀
#    필요: 환경변수 CF_ACCOUNT_ID·CF_API_TOKEN(cf_chat과 같은 것, 'D1 편집' 권한 추가) — 데이터베이스는 없으면 만듦(이름 querybot-chat)
#  - 내 컴퓨터 시험: ACCT_SQLITE=파일경로 를 주면 SQLite 파일에 같은 표로 저장
#  - 이메일·이름·사진은 저장하지 않음 (로그인 표 안에만)
import base64
import gzip
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request

DB_NAME = os.environ.get("ACCT_D1_NAME", "querybot-chat")
MAX_STORE = 1_900_000  # D1 한 칸 최대 2MB → gzip+base64 뒤 크기 상한
TABLE_SQL = ("CREATE TABLE IF NOT EXISTS acct (uid TEXT PRIMARY KEY, data TEXT NOT NULL, "
             "updated INTEGER NOT NULL, size INTEGER NOT NULL)")

_lock = threading.Lock()
_state = {"ready": False, "kind": "", "db": "", "reason": "준비 전", "checked": 0.0}
_sqlite = None


class StoreError(Exception):
    pass


class TooBig(StoreError):
    pass


def _cf():
    return os.environ.get("CF_ACCOUNT_ID", "").strip(), os.environ.get("CF_API_TOKEN", "").strip()


def _api(method: str, path: str, body=None, timeout: float = 20):
    acc, tok = _cf()
    url = f"https://api.cloudflare.com/client/v4/accounts/{acc}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        code = e.code
        try:
            msg = json.loads(e.read().decode("utf-8")).get("errors", [{}])[0].get("message", "")
        except Exception:  # noqa: BLE001
            msg = ""
        raise StoreError(f"http {code} {msg}"[:120])
    except Exception as e:  # noqa: BLE001
        raise StoreError(type(e).__name__)


def _d1(sql: str, params=None):
    j = _api("POST", f"/d1/database/{_state['db']}/query", {"sql": sql, "params": params or []})
    res = (j.get("result") or [{}])[0]
    if not j.get("success") or not res.get("success", True):
        raise StoreError("query failed")
    return res.get("results") or []


def setup() -> dict:
    """저장소 준비 (켤 때 한 번, 안 되면 2분마다 다시 — 권한을 나중에 추가해도 배포 없이 켜짐)"""
    global _sqlite
    with _lock:
        _state["checked"] = time.time()
        path = os.environ.get("ACCT_SQLITE", "").strip()
        if path:
            if _sqlite is None:
                _sqlite = sqlite3.connect(path, check_same_thread=False)
                _sqlite.execute(TABLE_SQL)
                _sqlite.commit()
            _state.update(ready=True, kind="sqlite", reason="")
            return dict(_state)
        acc, tok = _cf()
        if not acc or not tok:
            _state.update(ready=False, reason="CF_ACCOUNT_ID·CF_API_TOKEN 없음")
            return dict(_state)
        try:
            found = _api("GET", f"/d1/database?name={DB_NAME}")
            dbs = [d for d in (found.get("result") or []) if d.get("name") == DB_NAME]
            if dbs:
                _state["db"] = dbs[0]["uuid"]
            else:
                made = _api("POST", "/d1/database", {"name": DB_NAME})
                _state["db"] = (made.get("result") or {})["uuid"]
            _d1(TABLE_SQL)
            _state.update(ready=True, kind="d1", reason="")
        except StoreError as e:
            reason = str(e)
            if reason.startswith("http 401") or reason.startswith("http 403"):
                reason = "Cloudflare 열쇠에 'D1 편집' 권한이 없음"
            _state.update(ready=False, reason=reason)
        return dict(_state)


def status() -> dict:
    if not _state["ready"] and time.time() - _state["checked"] > 120:
        threading.Thread(target=setup, daemon=True).start()
    return {"ready": _state["ready"], "kind": _state["kind"], "reason": _state["reason"]}


def ready() -> bool:
    return bool(status()["ready"])


def _pack(obj) -> str:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    s = base64.b64encode(gzip.compress(raw, 6)).decode("ascii")
    if len(s) > MAX_STORE:
        raise TooBig(f"{len(s)}")
    return s


def _unpack(s: str):
    return json.loads(gzip.decompress(base64.b64decode(s)).decode("utf-8"))


def load(uid: str):
    """→ (묶음 또는 None, 저장 시각 ms)"""
    if not ready():
        raise StoreError("not ready")
    if _state["kind"] == "sqlite":
        with _lock:
            row = _sqlite.execute("SELECT data, updated FROM acct WHERE uid=?", (uid,)).fetchone()
        return (_unpack(row[0]), row[1]) if row else (None, 0)
    rows = _d1("SELECT data, updated FROM acct WHERE uid=?1", [uid])
    return (_unpack(rows[0]["data"]), int(rows[0]["updated"])) if rows else (None, 0)


def save(uid: str, obj) -> int:
    if not ready():
        raise StoreError("not ready")
    s = _pack(obj)
    now = int(time.time() * 1000)
    sql = ("INSERT INTO acct (uid, data, updated, size) VALUES (?1, ?2, ?3, ?4) "
           "ON CONFLICT(uid) DO UPDATE SET data=excluded.data, updated=excluded.updated, size=excluded.size")
    if _state["kind"] == "sqlite":
        with _lock:
            _sqlite.execute(sql.replace("?1", "?").replace("?2", "?").replace("?3", "?").replace("?4", "?"), (uid, s, now, len(s)))
            _sqlite.commit()
    else:
        _d1(sql, [uid, s, now, len(s)])
    return now


def delete(uid: str) -> None:
    if not ready():
        raise StoreError("not ready")
    if _state["kind"] == "sqlite":
        with _lock:
            _sqlite.execute("DELETE FROM acct WHERE uid=?", (uid,))
            _sqlite.commit()
    else:
        _d1("DELETE FROM acct WHERE uid=?1", [uid])
