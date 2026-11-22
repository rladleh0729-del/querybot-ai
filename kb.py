"""쿼리봇 지식 창고 (2026-11-22 사용자: "어떤 서버에 우리가 여러 파일들을 준비시켜 놓고 모델이 필요할 때 거기서 먼저 검색해보고 없으면 인터넷도"
→ "수능 수학 기출부터").

- 자료: 한국교육과정평가원 수능 수학 기출(문항 본문·보기·그림 설명·정답·배점). 원문은 공개 GitHub에 두지 않고
  Cloudflare D1(데이터베이스 querybot-chat, 표 kb)에만 둔다 — 서버는 처음 쓸 때 한 번 읽어 메모리에 두고 6시간마다 다시 읽음
- lookup(글): ① "2025학년도 수능 수학 1번"·"작년 수능 수학 21번"·"2027 수능 미적분 28번" 같은 번호 말 → 그 문항
               ② 번호가 없으면 글자(두 글자 묶음)가 많이 겹치는 문항 (search=True일 때만)
- 내 컴퓨터 시험: KB_LOCAL_JSON=폴더 를 주면 그 폴더의 kb_*.json을 읽음
"""
import json
import math
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

DB_NAME = os.environ.get("ACCT_D1_NAME", "querybot-chat")
SRC_URL = "https://www.suneung.re.kr/boardCnts/list.do?boardID=1500234&m=0403&s=suneung"
REFRESH = 6 * 3600
PARTS = {"공통": "공통", "확률과 통계": "확률과 통계", "확률과통계": "확률과 통계", "확통": "확률과 통계",
         "미적분": "미적분", "미적": "미적분", "기하": "기하"}
_lock = threading.Lock()
_st = {"items": [], "loaded": 0.0, "db": "", "err": ""}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [kb] {msg}", flush=True)


def _cf(method: str, path: str, body=None):
    acc, tok = os.environ.get("CF_ACCOUNT_ID", "").strip(), os.environ.get("CF_API_TOKEN", "").strip()
    req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{acc}{path}",
                                 data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _load() -> list:
    local = os.environ.get("KB_LOCAL_JSON", "").strip()
    if local:
        out = []
        for f in sorted(Path(local).glob("kb_*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            for it in d.get("items") or []:
                out.append(dict(it, year=d["year"], exam="수능"))
        return out
    if not _st["db"]:
        found = _cf("GET", f"/d1/database?name={DB_NAME}")
        dbs = [x for x in found.get("result") or [] if x.get("name") == DB_NAME]
        if not dbs:
            raise RuntimeError("D1 데이터베이스 없음")
        _st["db"] = dbs[0]["uuid"]
    j = _cf("POST", f"/d1/database/{_st['db']}/query", {"sql": "SELECT year, exam, part, no, points, text, choices, figure, answer FROM kb", "params": []})
    rows = (j.get("result") or [{}])[0].get("results") or []
    for r in rows:
        try:
            r["choices"] = json.loads(r.get("choices") or "null")
        except ValueError:
            r["choices"] = None
    return rows


def items() -> list:
    now = time.time()
    if _st["items"] and now - _st["loaded"] < REFRESH:
        return _st["items"]
    with _lock:
        if _st["items"] and now - _st["loaded"] < REFRESH:
            return _st["items"]
        if _st["err"] and now - _st["loaded"] < 300:  # 실패하면 5분 동안은 다시 안 함
            return _st["items"]
        try:
            its = _load()
            for it in its:
                it["_grams"] = _grams(f"{it['year']}학년도 수능 수학 {it['part']} {it['no']}번 " + (it.get("text") or "") + " " + (it.get("figure") or ""))
            _st.update(items=its, loaded=now, err="")
            log(f"지식 창고 {len(its)}문항 읽음")
        except Exception as e:  # noqa: BLE001 — 창고가 안 돼도 답은 그대로(인터넷 검색으로)
            _st.update(loaded=now, err=type(e).__name__)
            log(f"지식 창고 읽기 실패: {type(e).__name__} {str(e)[:80]}")
    return _st["items"]


def _grams(s: str) -> set:
    s = re.sub(r"\s+", "", re.sub(r"[$\\{}^_()]", " ", s or "")).lower()
    return {s[i:i + 2] for i in range(len(s) - 1)}


REF = re.compile(r"(?:(20\d\d|\d\d)\s*(학년도|년)?|(올해|이번|금년|작년|지난해|재작년))\s*(?:도\s*)?(?:대학수학능력시험|수능)"
                 r"[^\d]{0,24}?(?:수학|수리)?[^\d]{0,12}?(공통|확률과\s?통계|확통|미적분|미적|기하)?[^\d]{0,6}?(\d{1,2})\s*번")


def _years(m, have: list) -> list:
    if m.group(3):
        latest = max(have) if have else 0
        off = {"올해": 0, "이번": 0, "금년": 0, "작년": 1, "지난해": 1, "재작년": 2}[m.group(3)]
        return [latest - off]
    y = int(m.group(1))
    y = y + 2000 if y < 100 else y
    if m.group(2) == "년":  # 'N년 수능'은 보통 N학년도를 뜻하지만, N년에 본 시험(N+1학년도)일 수도 있음
        return [y, y + 1]
    return [y]


def lookup(text: str, search: bool = False, top: int = 2) -> list:
    """맞는 문항 목록(최대 top개, 번호로 찾으면 선택과목 셋까지). 없으면 빈 목록."""
    its = items()
    if not its or not text:
        return []
    have = sorted({it["year"] for it in its})
    out = []
    for m in REF.finditer(text):
        if not re.search(r"수학|수리|확률|확통|미적|기하", m.group(0)):  # 국어·영어 등 다른 과목 번호는 안 찾음
            continue
        pm = re.search(r"확률과\s?통계|확통|미적분|미적|기하|공통", m.group(0))  # 정규식 안의 선택 묶음은 건너뛰어질 수 있어 따로 찾음
        part = PARTS.get(pm.group(0).replace(" ", "")) if pm else None
        no = int(m.group(5))
        for y in _years(m, have):
            hit = [it for it in its if it["year"] == y and it["no"] == no and (it["part"] == part if part else (it["part"] == "공통" if no <= 22 else True))]
            if hit:
                out += hit
                break
    if out or not search:
        return out[:4]
    q = _grams(text)
    if len(q) < 4:
        return []
    scored = []
    for it in its:
        inter = len(q & it["_grams"])
        if inter:
            scored.append((inter / math.sqrt(len(q) * len(it["_grams"])), it))
    scored.sort(key=lambda x: -x[0])
    return [it for s, it in scored[:top] if s >= 0.12]


def note(hits: list) -> str:
    """모델에게 줄 자료 글"""
    parts = []
    for it in hits:
        ch = it.get("choices")
        s = (f"[쿼리봇 자료 · 한국교육과정평가원 {it['year']}학년도 대학수학능력시험 수학 영역 "
             f"{'' if it['part'] == '공통' else it['part'] + ' '}{it['no']}번{' (' + str(it['points']) + '점)' if it.get('points') else ''}]\n"
             f"문제: {it.get('text') or ''}")
        if ch:
            s += "\n보기: " + "  ".join(ch)
        if it.get("figure"):
            s += f"\n그림 설명: {it['figure']}"
        if it.get("answer"):
            s += f"\n공식 정답: {it['answer']}" + ("번" if ch else "")
        parts.append(s)
    return ("아래는 쿼리봇 지식 창고에 있는 평가원 기출 원문이야(인터넷 검색보다 정확함). 이걸 바탕으로 답하고, 문제를 보여 줄 땐 수식을 $...$로 그대로 써. "
            "공식 정답을 그대로 따르고, 풀이를 물으면 단계별로 풀어 그 정답에 닿게 해. 끝에 '출처: 한국교육과정평가원'을 밝혀.\n\n" + "\n\n".join(parts))


def sources(hits: list) -> list:
    return [{"title": f"{it['year']}학년도 수능 수학 {'' if it['part'] == '공통' else it['part'] + ' '}{it['no']}번 (한국교육과정평가원)", "url": SRC_URL} for it in hits]
