"""웹 검색 (2026-11-09 사용자 요청: "검색 기능도") — Gemma 4가 <<search: 검색어>>를 내면 cf_chat이 부른다.

- 기본: Bing 검색 페이지(열쇠 없음, PC 0.3초·10건). DuckDuckGo는 Render에서 응답이 없어(2026-11-09) 예비로만.
- 환경변수 TAVILY_API_KEY(무료 월 1,000번) 또는 BRAVE_API_KEY가 있으면 그쪽을 먼저 쓴다.
- 결과 위 2개는 본문도 조금 읽어 온다(요약만으로는 답이 얕아서). 전체 4초 안팎.
- 같은 검색어는 10분 동안 기억(같은 질문 반복에 검색을 또 하지 않게).
"""
import base64
import concurrent.futures
import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
BRAVE_KEY = os.environ.get("BRAVE_API_KEY", "").strip()
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
MAX_RESULTS = 5
PAGE_CHARS = 1200      # 본문을 읽어 올 때 결과 하나에 붙일 글자 수
PAGE_READ = 2          # 위에서 몇 개까지 본문을 읽을지
CACHE_SEC = 600
_cache: dict = {}
_lock = threading.Lock()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [search] {msg}", flush=True)


def _strip(s: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s or "")).split())


def _ddg(q: str) -> list:
    data = urllib.parse.urlencode({"q": q, "kl": "kr-kr"}).encode()
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=data,
                                 headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=5) as r:
        b = r.read().decode("utf-8", "replace")
    out = []
    for m in re.finditer(r'<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>', b, re.S):
        u = html.unescape(m.group(1))
        g = re.search(r"uddg=([^&]+)", u)
        if g:
            u = urllib.parse.unquote(g.group(1))
        if not u.startswith("http") or "duckduckgo.com/y.js" in u:  # 광고 줄은 뺀다
            continue
        out.append({"title": _strip(m.group(2))[:120], "url": u, "snippet": _strip(m.group(3))[:400]})
    if not out and ("anomaly" in b or "captcha" in b.lower()):
        raise RuntimeError("DuckDuckGo 차단(봇 확인)")
    return out


def _bing(q: str) -> list:
    """Bing 검색 페이지(열쇠 없음). Render(해외 클라우드)에서 DuckDuckGo가 응답하지 않아(8초 시간 초과, 2026-11-09) 기본으로 씀."""
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": q, "setlang": "ko", "cc": "KR", "count": 10})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8", "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=6) as r:
        b = r.read().decode("utf-8", "replace")
    out = []
    for blk in re.findall(r'<li class="b_algo"(.*?)</li>', b, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        u = html.unescape(m.group(1))
        g = re.search(r"[?&]u=a1([^&]+)", u)
        if "bing.com/ck/a" in u and g:  # 클릭 추적 주소면 원래 주소를 꺼낸다
            s = g.group(1) + "=" * (-len(g.group(1)) % 4)
            try:
                u = base64.urlsafe_b64decode(s).decode("utf-8")
            except Exception:  # noqa: BLE001
                continue
        if not u.startswith("http"):
            continue
        sn = re.search(r"<p[^>]*>(.*?)</p>", blk, re.S)
        out.append({"title": _strip(m.group(2))[:120], "url": u, "snippet": _strip(sn.group(1) if sn else "")[:400]})
    if not out and ("captcha" in b.lower() or len(b) < 20000):
        raise RuntimeError(f"Bing 결과 없음({len(b)}바이트)")
    return out


def _tavily(q: str) -> list:
    body = json.dumps({"api_key": TAVILY_KEY, "query": q, "max_results": MAX_RESULTS, "search_depth": "basic"}).encode()
    req = urllib.request.Request("https://api.tavily.com/search", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=12) as r:
        d = json.loads(r.read().decode("utf-8"))
    return [{"title": (x.get("title") or "")[:120], "url": x.get("url") or "", "snippet": (x.get("content") or "")[:900], "full": True}
            for x in d.get("results") or [] if (x.get("url") or "").startswith("http")]


def _brave(q: str) -> list:
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": MAX_RESULTS, "country": "KR", "search_lang": "ko"})
    req = urllib.request.Request(url, headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    return [{"title": _strip(x.get("title"))[:120], "url": x.get("url") or "", "snippet": _strip(x.get("description"))[:400]}
            for x in (d.get("web") or {}).get("results") or [] if (x.get("url") or "").startswith("http")]


def _page_text(url: str) -> str:
    """결과 페이지 본문 조금: 스크립트·메뉴 같은 것은 빼고 글자만."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ko,en;q=0.8"})
        with urllib.request.urlopen(req, timeout=4) as r:
            if "html" not in (r.headers.get("Content-Type") or ""):
                return ""
            raw = r.read(400_000)
            cs = r.headers.get_content_charset() or "utf-8"
        b = raw.decode(cs, "replace")
    except Exception:  # noqa: BLE001 — 못 읽으면 요약만 쓴다
        return ""
    b = re.sub(r"(?is)<(script|style|noscript|nav|header|footer|aside|form|svg)[^>]*>.*?</\1>", " ", b)
    m = re.search(r"(?is)<(article|main)[^>]*>(.*?)</\1>", b)
    if m:
        b = m.group(2)
    lines = [ln for ln in (_strip(x) for x in re.split(r"(?i)</(p|div|li|h[1-6]|tr|br)\s*>|<br\s*/?>", b) if x) if len(ln) > 25]
    return " ".join(lines)[:PAGE_CHARS]


def search(q: str) -> list:
    """[{title, url, snippet}] 최대 5개. 다 실패하면 빈 목록."""
    q = " ".join((q or "").split())[:120]
    if not q:
        return []
    with _lock:
        hit = _cache.get(q)
        if hit and time.time() - hit[0] < CACHE_SEC:
            return hit[1]
    t0 = time.time()
    found, used = [], ""
    for name, fn, ok in (("tavily", _tavily, TAVILY_KEY), ("brave", _brave, BRAVE_KEY), ("bing", _bing, True), ("ddg", _ddg, True)):
        if not ok:
            continue
        try:
            found = fn(q)[:MAX_RESULTS]
            used = name
            if found:
                break
        except Exception as e:  # noqa: BLE001
            log(f"{name} 실패: {type(e).__name__} {str(e)[:80]}")
    # 위 결과 몇 개는 본문도 읽어 온다 (Tavily는 이미 본문 요약을 줌)
    need = [x for x in found[:PAGE_READ] if not x.get("full")]
    if need:
        with concurrent.futures.ThreadPoolExecutor(len(need)) as ex:
            for x, txt in zip(need, ex.map(lambda r: _page_text(r["url"]), need)):
                if txt:
                    x["page"] = txt
    for x in found:
        x.pop("full", None)
    log(f"검색어 {len(q)}자 → {used or '없음'} {len(found)}건 ({time.time() - t0:.1f}초)")  # 방문자 말은 로그에 남기지 않는다
    if found:
        with _lock:
            _cache[q] = (time.time(), found)
            for k in [k for k, v in _cache.items() if time.time() - v[0] > CACHE_SEC]:
                del _cache[k]
    return found
