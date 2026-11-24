"""쿼리봇 지식 창고 (2026-11-22 사용자: "어떤 서버에 우리가 여러 파일들을 준비시켜 놓고 모델이 필요할 때 거기서 먼저 검색해보고 없으면 인터넷도"
→ "수능 수학 기출부터" → 2026-11-24 "이제 좀 여러 자료들을 골고루 넣어 줘").

- 자료: 한국교육과정평가원 수능 기출 2025~2027학년도 — 국어(공통·화법과 작문·언어와 매체), 수학(공통·확률과 통계·미적분·기하),
  영어(듣기 대본 포함), 한국사, 사회탐구 9과목, 과학탐구 8과목. 문항 본문·보기·그림 설명·묶음 지문·정답·배점.
  원문은 공개 GitHub에 두지 않고 Cloudflare D1(데이터베이스 querybot-chat, 표 kb)에만 둔다 — 서버는 처음 쓸 때 한 번 읽어 메모리에 두고 6시간마다 다시 읽음
- lookup(글): ① "2025학년도 수능 국어 5번"·"작년 수능 영어 31번"·"2027 수능 생윤 3번"·"2026 수능 물리1 20번" 같은 번호 말 → 그 문항
               ② 번호가 없으면 검색어의 드문 낱말 조각이 많이 들어 있는 문항 (search=True이고 검색어가 기출을 말할 때만)
- 내 컴퓨터 시험: KB_LOCAL_JSON=폴더 를 주면 그 폴더의 *.json(items 목록)을 읽음
"""
import json
import math
import os
import re
import threading
import time
import urllib.request
from array import array
from collections import Counter
from pathlib import Path

DB_NAME = os.environ.get("ACCT_D1_NAME", "querybot-chat")
SRC_URL = "https://www.suneung.re.kr/boardCnts/list.do?boardID=1500234&m=0403&s=suneung"
REFRESH = 6 * 3600
_lock = threading.Lock()
_st = {"items": [], "post": {}, "loaded": 0.0, "db": "", "err": ""}

# 과목 이름·줄임말 → (영역, 과목). 위에서부터 먼저 잡고, 겹치는 글자는 한 번만 씀(물리학Ⅱ를 물리학Ⅰ보다 먼저 등)
_N = r"(?!\s*번)"  # '물리 2번'의 2는 Ⅱ가 아니라 문항 번호
SUBJ = [(re.compile(p), a, s) for p, a, s in [
    (r"화법과\s?작문|화작", "국어", "화법과 작문"),
    (r"언어와\s?매체|언매", "국어", "언어와 매체"),
    (r"국어|언어\s?영역", "국어", None),
    (r"확률과\s?통계|확통", "수학", "확률과 통계"),
    (r"미적분|미적", "수학", "미적분"),
    (r"기하", "수학", "기하"),
    (r"수학|수리", "수학", None),
    (r"공통", None, "공통"),
    (r"영어|외국어", "영어", None),
    (r"한국사|국사", "한국사", None),
    (r"생활과\s?윤리|생윤", "사회탐구", "생활과 윤리"),
    (r"윤리와\s?사상|윤사", "사회탐구", "윤리와 사상"),
    (r"한국\s?지리|한지", "사회탐구", "한국지리"),
    (r"세계\s?지리|세지", "사회탐구", "세계지리"),
    (r"동아시아사", "사회탐구", "동아시아사"),
    (r"세계사", "사회탐구", "세계사"),
    (r"정치와\s?법|정법", "사회탐구", "정치와 법"),
    (r"사회\s?[·･․ㆍ・]?\s?문화|사문", "사회탐구", "사회·문화"),
    (r"경제", "사회탐구", "경제"),
    (r"사회\s?탐구|사탐", "사회탐구", None),
    (r"(?:물리학?|물)\s?(?:Ⅱ|II|2(?!\d))" + _N, "과학탐구", "물리학Ⅱ"),
    (r"(?:화학|화)\s?(?:Ⅱ|II|2(?!\d))" + _N, "과학탐구", "화학Ⅱ"),
    (r"(?:생명\s?과학|생명|생물|생)\s?(?:Ⅱ|II|2(?!\d))" + _N, "과학탐구", "생명과학Ⅱ"),
    (r"(?:지구\s?과학|지구|지학|지)\s?(?:Ⅱ|II|2(?!\d))" + _N, "과학탐구", "지구과학Ⅱ"),
    (r"(?:물리학?|물)\s?(?:Ⅰ|I|1(?!\d))" + _N + r"|물리학?", "과학탐구", "물리학Ⅰ"),
    (r"(?:화학|화)\s?(?:Ⅰ|I|1(?!\d))" + _N + r"|화학", "과학탐구", "화학Ⅰ"),
    (r"(?:생명\s?과학|생명|생물|생)\s?(?:Ⅰ|I|1(?!\d))" + _N + r"|생명\s?과학|생물", "과학탐구", "생명과학Ⅰ"),
    (r"(?:지구\s?과학|지구|지학|지)\s?(?:Ⅰ|I|1(?!\d))" + _N + r"|지구\s?과학|지학", "과학탐구", "지구과학Ⅰ"),
    (r"과학\s?탐구|과탐", "과학탐구", None),
]]
EXAM = re.compile(r"(?:(20\d\d|\d\d)\s*(학년도|년)?|(올해|이번|금년|작년|지난해|재작년))\s*(?:도\s*)?(?:대학수학능력시험|수능)")
NO = re.compile(r"(\d{1,2})\s*번")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [kb] {msg}", flush=True)


def _cf(method: str, path: str, body=None):
    acc, tok = os.environ.get("CF_ACCOUNT_ID", "").strip(), os.environ.get("CF_API_TOKEN", "").strip()
    req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{acc}{path}",
                                 data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _load() -> list:
    local = os.environ.get("KB_LOCAL_JSON", "").strip()
    if local:
        out = []
        for f in sorted(Path(local).glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            for it in d.get("items") or []:
                out.append(dict(it, year=d["year"], exam="수능", area=d.get("area") or "수학"))
        return out
    if not _st["db"]:
        found = _cf("GET", f"/d1/database?name={DB_NAME}")
        dbs = [x for x in found.get("result") or [] if x.get("name") == DB_NAME]
        if not dbs:
            raise RuntimeError("D1 데이터베이스 없음")
        _st["db"] = dbs[0]["uuid"]
    rows, off = [], 0
    while True:  # 지문이 길어 한 번에 다 받지 않고 300줄씩
        j = _cf("POST", f"/d1/database/{_st['db']}/query",
                {"sql": "SELECT year, exam, area, part, no, points, text, choices, figure, answer, passage, prange FROM kb ORDER BY id LIMIT 300 OFFSET ?", "params": [off]})
        got = (j.get("result") or [{}])[0].get("results") or []
        rows += got
        if len(got) < 300:
            break
        off += 300
    for r in rows:
        r["area"] = r.get("area") or "수학"
        try:
            r["choices"] = json.loads(r.get("choices") or "null")
        except ValueError:
            r["choices"] = None
    return rows


def _label(it) -> str:
    p = it.get("part") or "공통"
    return f"{it['area']} 영역 " + ("" if p == "공통" else p + " ") + f"{it['no']}번"


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
            post = {}  # 낱말 조각 → 문항 번호들 (문항마다 조각 집합을 들고 있으면 메모리가 커서 거꾸로 색인)
            for i, it in enumerate(its):
                for x in _toks((it.get("text") or "") + " " + " ".join(it.get("choices") or []) + " " + (it.get("figure") or "") + " " + (it.get("passage") or "")):
                    a = post.get(x)
                    if a is None:
                        a = post[x] = array("H")
                    a.append(i)
            _st.update(items=its, post=post, loaded=now, err="")
            log(f"지식 창고 {len(its)}문항 읽음")
        except Exception as e:  # noqa: BLE001 — 창고가 안 돼도 답은 그대로(인터넷 검색으로)
            _st.update(loaded=now, err=type(e).__name__)
            log(f"지식 창고 읽기 실패: {type(e).__name__} {str(e)[:80]}")
    return _st["items"]


def _toks(s: str) -> set:
    """찾기용 조각: 한글은 띄어쓰기를 안 넘는 두 글자 묶음, 영어는 세 글자 이상 낱말, 숫자는 덩어리"""
    s = (s or "").lower()
    out = {"w:" + w for w in re.findall(r"[a-z]{3,}", s)}
    out.update("n:" + n for n in re.findall(r"\d+(?:[./^]\d+)*", s) if len(n) > 1)
    for run in re.findall(r"[가-힣]{2,}", s):
        out.update(run[i:i + 2] for i in range(len(run) - 1))
    return out


# 글자로 찾을 때 검색어에서 빼는 말(시험 이름·요청 말) — 문항 내용과 상관없어 점수를 흐림
STOP = re.compile(r"대학수학능력시험|수능|평가원|기출|학년도|모의고사|모의평가|문제|문항|지문|풀이|해설|정답|영역|과목|관련|내용|"
                  r"알려\s?줘|보여\s?줘|찾아\s?줘|나온|나오는|\d+\s*번|20\d\d")


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


def _subject(win: str):
    """'수능' 뒤 짧은 글에서 (영역, 과목, 번호). 과목 이름 속 숫자(물1·화2)는 번호로 안 봄"""
    taken = [False] * len(win)
    found = []
    for pat, area, part in SUBJ:
        for mm in pat.finditer(win):
            if any(taken[mm.start():mm.end()]):
                continue
            for i in range(mm.start(), mm.end()):
                taken[i] = True
            found.append((mm.start(), area, part))
    n = NO.search("".join(" " if taken[i] else ch for i, ch in enumerate(win)))
    if not n:
        return None
    found = sorted(f for f in found if f[0] < n.start())
    area = next((a for _, a, _p in found if a), None)
    part = next((p for _, a, p in found if p and (a is None or a == area)), None)
    if not area:
        return None
    return area, part, int(n.group(1))


def lookup(text: str, search: bool = False, top: int = 2) -> list:
    """맞는 문항 목록(번호로 찾으면 선택과목 셋까지, 글자로 찾으면 최대 top개). 없으면 빈 목록."""
    its = items()
    if not its or not text:
        return []
    have = sorted({it["year"] for it in its})
    out = []
    marks = list(EXAM.finditer(text))
    for k, m in enumerate(marks):
        end = min(m.end() + 40, marks[k + 1].start() if k + 1 < len(marks) else len(text))
        sub = _subject(text[m.end():end])
        if not sub:  # 과목을 안 말하면(예: '2025 수능 1번') 어느 과목인지 몰라 안 찾음
            continue
        area, part, no = sub
        for y in _years(m, have):
            cand = [it for it in its if it["year"] == y and it["area"] == area and it["no"] == no]
            if part:
                cand = [it for it in cand if it["part"] == part]
            elif len({it["part"] for it in cand}) > 1:
                com = [it for it in cand if it["part"] == "공통"]
                cand = com or (cand if len(cand) <= 3 else [])  # 과탐 '3번'처럼 과목을 모르면 안 찾음
            if cand:
                out += cand
                break
    if out or not search:
        return out[:4]
    # 2026-11-24 사용자: "다른 때도 갑자기 수능 문제를 풀어주고 다시 말해도 계속 풀이하는 괴상한 병"
    # → 웹 검색어마다 글자 겹침으로 뒤지다 '2026 월드컵 일정'·'2026년 11월 뉴스'·'등차수열 합 공식'까지 기출 문항에 걸려 웹 검색을 가로챘음.
    #   글자 겹침 찾기는 검색어가 기출을 말할 때만(수능·대학수학능력시험·평가원·기출·학년도·모의고사)
    if not re.search(r"수능|대학수학능력시험|평가원|기출|학년도|모의고사", text):
        return []
    # 2026-11-24 여러 과목·긴 지문이 들어와 방식 바꿈: 예전(두 글자 겹침 비율 + '2025학년도 수능 수학' 이름표)은
    # '2025 수능 영어 …' 검색에 이름표만 겹친 짧은 수학 문항이 걸렸음 → 이름표는 빼고 학년도·과목은 거르는 조건으로,
    # 점수는 드문 조각일수록 무겁게(idf) 센 '검색어 조각이 그 문항에 들어 있는 비율' — 창고에 아예 없는 조각도 분모에 넣어 엉뚱한 검색은 안 걸림
    years = {y for m in EXAM.finditer(text) for y in _years(m, have)} | {int(y) for y in re.findall(r"20\d\d", text) if int(y) in have}
    areas, parts, taken = set(), set(), [False] * len(text)
    for pat, area, part in SUBJ:
        for mm in pat.finditer(text):
            if any(taken[mm.start():mm.end()]):
                continue
            for i in range(mm.start(), mm.end()):
                taken[i] = True
            if part and part != "공통" and len(mm.group(0).replace(" ", "")) >= 3:
                parts.add(part)
            elif not part and area:
                areas.add(area)
    q = _toks(STOP.sub(" ", "".join(" " if taken[i] else ch for i, ch in enumerate(text))))
    if len(q) < 2:
        return []
    post, n = _st["post"], len(its)
    oks = {i for i, it in enumerate(its) if (not years or it["year"] in years) and (not areas or it["area"] in areas) and (not parts or it["part"] in parts)}
    if not oks:
        return []
    w = {x: math.log((n + 1) / (len(post[x]) + 0.5)) if x in post else math.log(2 * (n + 1)) for x in q}
    total = sum(w.values()) or 1.0
    cnt = Counter()
    for x in q:
        for i in post.get(x, ()):
            if i in oks:
                cnt[i] += w[x]
    scored = sorted(((c / total, its[i]) for i, c in cnt.items()), key=lambda x: -x[0])
    return [it for s, it in scored[:top] if s >= 0.6]


def note(hits: list) -> str:
    """모델에게 줄 자료 글 (묶음 지문은 한 번만)"""
    parts, seen = [], set()
    for it in hits:
        ch = it.get("choices")
        s = (f"[쿼리봇 자료 · 한국교육과정평가원 {it['year']}학년도 대학수학능력시험 {_label(it)}"
             f"{' (' + str(it['points']) + '점)' if it.get('points') else ''}]\n")
        if it.get("passage"):
            key = (it["year"], it["area"], it.get("part"), it.get("prange"))
            if key in seen:
                s += f"(지문은 위와 같음)\n"
            else:
                seen.add(key)
                s += ("듣기 대본: " if it.get("prange") == "듣기 대본" else f"지문 [{it.get('prange') or ''}]: ") + it["passage"] + "\n"
        s += f"문제: {it.get('text') or ''}"
        if ch:
            s += "\n보기: " + "  ".join(ch)
        if it.get("figure"):
            s += f"\n그림·자료 설명: {it['figure']}"
        if it.get("answer"):
            s += f"\n공식 정답: {it['answer']}" + ("번" if ch else "")
        parts.append(s)
    return ("아래는 쿼리봇 지식 창고에 있는 평가원 기출 원문이야(인터넷 검색보다 정확함). 사용자의 이번 말이 이 문항에 관한 것일 때만 쓰고, "
            "상관없는 질문이면 이 자료는 무시하고 문제를 풀지 말고 평소처럼 답해. 이 문항에 관한 것이면 이걸 바탕으로 답하고, 문제를 보여 줄 땐 원문을 그대로(수식은 $...$로) 써. "
            "공식 정답을 그대로 따르고, 풀이·해설을 물으면 단계별로 근거를 들어 그 정답에 닿게 해. 끝에 '출처: 한국교육과정평가원'을 밝혀.\n\n" + "\n\n".join(parts))


def sources(hits: list) -> list:
    return [{"title": f"{it['year']}학년도 수능 {_label(it).replace(' 영역', '')} (한국교육과정평가원)", "url": SRC_URL} for it in hits]
