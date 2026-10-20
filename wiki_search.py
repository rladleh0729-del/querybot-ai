"""지식 질문이면 한국어 위키백과에서 찾아 짧은 요약을 돌려준다. (2026-10-20, 사용자 요청)

  lookup(text) -> {"reply": 답, "title": 문서 제목, "url": 문서 주소} 또는 None(지식 질문 아님 / 못 찾음)

- 작은 모델은 글을 읽고 요약하는 법을 배우지 않아서, 검색 결과를 모델에 넣지 않고 첫 1~2문장을 그대로 보여 준다.
- 위키백과 글은 CC BY-SA라 답에 출처(문서 제목)를 붙이고, 주소는 url로 따로 돌려준다.
- 표준 라이브러리만 쓴다 (PC 서버 ai_server.py와 클라우드 ai_cloud/app.py가 같은 파일을 씀 — 고치면 둘 다 복사).
"""
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict

API = "https://ko.wikipedia.org/w/api.php"
REST = "https://ko.wikipedia.org/api/rest_v1/page/summary/"
UA = "QueryBotAI/1.0 (https://querybot.kr/ai)"  # 위키미디어는 알아볼 수 있는 User-Agent를 요구한다
TIMEOUT = 4
MAX_CHARS = 220

ASK = r"(누구야|누구예요|누구지|누군데|누구니|누구인가요|누구임|뭐야|뭐예요|뭐지|뭔데|뭐니|뭐임|무엇인가요|무엇이야|어디야|어디예요|어디지|어딘데)"
PLEASE = r"(줘|줄래|주세요|줘요|줄래요|줄 수 있어)"
PATTERNS = [
    ("capital", re.compile(r"^(?P<e>.+?)의?\s*수도(는|가|가 어디야|는 어디야|는 어디예요|가 뭐야|는 뭐야)?$")),
    ("about", re.compile(rf"^(?P<e>.+?)\s*(에 대해서?|에 관해서?)\s*(좀\s*)?(알려|설명해|말해|이야기해)\s*{PLEASE}?$")),
    ("about", re.compile(rf"^(?P<e>.+?)\s+(좀\s*)?(설명해|알려)\s*{PLEASE}$")),
    ("what", re.compile(rf"^(?P<e>.+?)(은|는|이|가)\s*{ASK}$")),
    ("what", re.compile(r"^(?P<e>.+?)(이란|란)\s*(뭐야|뭐예요|무엇인가요|무엇이야)?$")),
    ("know", re.compile(r"^(?P<e>.+?)\s*(을|를)?\s*알아$")),
]
# 백과사전에서 찾을 대상이 아닌 말 (나·너, 오늘, 이름·기분 같은 대화용 말)
SELF = re.compile(r"(^|\s)(너|넌|네|니|당신|쿼리봇|나|난|내|저|전|제|우리|이거|그거|저거|이게|그게|이건|그건|뭐|무슨|어떤|오늘|지금|내일|어제|여기|거기|이|그|저)(\s|$)")
NOT_TOPIC = {"이름", "네 이름", "취미", "정체", "기분", "꿈", "나이", "생일", "날씨", "시간", "몇 시", "요일", "날짜", "비밀", "정답",
             "문제", "이유", "방법", "거", "것", "답", "말", "뜻", "차이", "점심", "저녁", "아침", "메뉴", "다음", "할 일", "계획"}

_cache: "OrderedDict[str, tuple]" = OrderedDict()
_lock = threading.Lock()


def find_topic(text: str):
    """지식 질문이면 (종류, 대상) — 아니면 None."""
    t = re.sub(r"[?？!.~ㅎㅋ\s]+$", "", " ".join(text.split()))
    for kind, rx in PATTERNS:
        m = rx.match(t)
        if not m:
            continue
        e = m.group("e").strip(" ,'\"「」")
        if not (1 <= len(e) <= 30) or not re.search(r"[가-힣A-Za-z0-9]", e):
            return None
        if e in NOT_TOPIC or SELF.search(e):
            return None
        return kind, e
    return None


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _search(query: str, n: int = 3) -> list:
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": n,
                                "format": "json", "utf8": 1, "srprop": ""})
    return [h["title"] for h in _get(f"{API}?{q}").get("query", {}).get("search", [])]


def _summary(title: str):
    d = _get(REST + urllib.parse.quote(title.replace(" ", "_"), safe=""))
    if d.get("type") == "disambiguation" or not d.get("extract"):
        return None
    url = d.get("content_urls", {}).get("desktop", {}).get("page") or f"https://ko.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    return d.get("title", title), d["extract"], url


def _clean(text: str) -> str:
    # 괄호 속 원어·한자·발음(예: "(러시아어: Василий …)")은 읽기 어려워서 뺀다.
    text = text.replace(" ", " ")
    for _ in range(3):
        text = re.sub(r"\s*\([^()]*\)", "", text)
    # 요약에서 닫는 괄호가 빠진 경우: "이순신(李舜臣, … 16일 은" → "이순신은"
    text = re.sub(r"\s*\([^()]*?\s(?=(은|는|이|가)\s)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> list:
    return [x for x in re.split(r"(?<=[.!?])\s+", _clean(text)) if x]


def _short(extract: str) -> str:
    sents = _sentences(extract)
    out = ""
    for s in sents[:2]:
        if out and len(out) + len(s) > MAX_CHARS:
            break
        out = (out + " " + s).strip()
    return out if len(out) <= MAX_CHARS else out[:MAX_CHARS].rstrip() + "…"


def _norm(s: str) -> str:
    return re.sub(r"[\s()·_\-]", "", s).lower()


def _lookup(kind: str, e: str):
    if kind == "capital":
        key = f"{e}의 수도".replace(" ", "")
        for title in _search(f"{e}의 수도", 5):  # "울란바토르는 몽골의 수도이다" 같은 도시 문서를 먼저
            s = _summary(title)
            if s and key in "".join(_sentences(s[1])[:2]).replace(" ", ""):
                return s
        s = _summary(e) if _search(e, 1) else None  # 없으면 나라 문서에서 수도가 나오는 문장만
        if s:
            for sent in _sentences(s[1]):
                if "수도" in sent:
                    return s[0], sent, s[2]
        return None
    ne = _norm(e)
    for title in _search(e):
        nt = _norm(re.sub(r"\s*\(.*\)$", "", title))  # "파리 (프랑스)" → "파리"
        # 제목이 질문한 말과 맞을 때만 (엉뚱한 문서를 보여 주지 않게):
        # 같음 / 성·이름처럼 끝이 같음(칸딘스키→바실리 칸딘스키) / 질문이 더 김(세종대왕→세종) / 앞이 같고 조금 김(서울→서울특별시)
        ok = nt == ne or nt.endswith(ne) or nt in ne or (nt.startswith(ne) and len(nt) <= len(ne) + 4)
        if not ok:
            continue
        s = _summary(title)
        if s and not s[1].lstrip().startswith("《"):  # 드라마·책 같은 작품 문서는 빼기
            return s
    return None


def lookup(text: str):
    topic = find_topic(text)
    if not topic:
        return None
    key = f"{topic[0]}:{topic[1]}"
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < (86400 if hit[1] else 600):
            _cache.move_to_end(key)
            return hit[1]
    result = None
    try:
        s = _lookup(*topic)
        if s:
            title, extract, url = s
            result = {"reply": f"위키백과에서 찾아봤어요. {_short(extract)}\n(출처: 위키백과 「{title}」)", "title": title, "url": url}
    except Exception:
        result = None  # 인터넷이 안 되거나 위키백과가 느리면 그냥 모델 답으로
    with _lock:
        _cache[key] = (now, result)
        while len(_cache) > 500:
            _cache.popitem(last=False)
    return result


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for q in sys.argv[1:]:
        print(q, "→", find_topic(q), "→", lookup(q))
