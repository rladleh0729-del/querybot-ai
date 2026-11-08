"""Cloudflare Workers AI(Gemma 4 26B)로 답하기 (2026-11-08). app.py에서 방문자가 'Gemma 4'를 고르면 쓴다.

- 열쇠는 Render 환경변수에만: CF_ACCOUNT_ID, CF_API_TOKEN (없으면 ready()가 False → 목록에 안 나옴)
- Cloudflare 무료 요금제: 하루 10,000뉴런(한국 시각 오전 9시 초기화), 넘치면 오류만(요금 없음).
  PC 시험(cf_compare.py): 질문 1번에 약 $0.000041 → 무료로 하루 약 2,700번, 평균 2초, 사실 질문 12개 지어냄 0.
- 사실 질문(몇·언제·누가·어디…)은 위키백과 문서 2건을 '참고 자료'로 붙이고, 자료에 없으면 모른다고 하게 한다.
  (위키 없이 물으면 8B·Qwen 모두 숫자·이름을 지어냄 — 2026-11-08 시험)
- PC판 원본: C:\\python venv\\qwen8b_local\\qwen_server.py (같은 규칙)
"""
import datetime
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

import wiki_search

CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "").strip()
CF_MODEL = os.environ.get("CF_MODEL", "@cf/google/gemma-4-26b-a4b-it")
MAX_TOKENS = 600        # 수식 설명처럼 긴 답도 잘리지 않게(보통 답은 30~110토큰)
TEMPERATURE = 0.6
HISTORY_TURNS = 6       # 앞 대화는 최근 6개까지만
TIMEOUT = 60

SYSTEM_PROMPT = (
    "너는 '쿼리봇'이라는 한국어 대화 AI야. 이름을 물으면 쿼리봇이라고 답해. "
    "질문에 자연스럽고 친절하게 답해. 보통은 1~3문장으로 짧게, 설명이 필요한 질문은 필요한 만큼. "
    "숫자·날짜·고유명사처럼 확실하지 않은 사실은 지어내지 말고 '정확히는 몰라요'라고 말해. "
    "수식은 LaTeX로 써: 문장 속 수식은 $...$, 따로 보여 줄 수식은 $$...$$. 굵은 글씨(**)나 제목(#) 같은 마크다운 기호는 쓰지 마. "
    "사용자가 영어 등 외국어로 말해도 그대로 한국어로 친절히 답하고, 한국어로 말하라고 요구하지 마. "
    "답에 한자나 키릴·일본 글자는 섞지 마."
)
# 위키 참고 자료: Gemma 4는 자료 없이 더 잘 답하고(에펠탑 330m·백두산 2,744m 정답, 모르는 건 모른다고 함),
# 자료를 붙이면 "제공된 자료에 없어서 몰라요"처럼 오히려 멍청해 보여서 끔 (2026-11-08 시험·사용자 의견)
USE_WIKI_CONTEXT = False
FACT_Q = re.compile(r"(몇\s?(년|개|명|글자|자|번|살|층|미터|m|km|㎞)|언제|누가|누구|어디|어느 나라|인구|면적|태어난|태어났|사망|돌아가|"
                    r"발명|건립|세워진|설립|창제|만든 (해|사람)|길이|높이|무게|최초|처음 (만든|발견))")
SELF_Q = re.compile(r"(^|\s)(너|넌|네|니|당신|쿼리봇|나|난|내|저|제|우리)(는|가|이|의|를|도|한테|랑|야)?(\s|$|[?!.])")
STRAY = re.compile(r"[\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF]")
THINK = re.compile(r"<think>.*?</think>", re.S)


# ---------------------------------------------------------
# 오늘 쓴 뉴런 (무료 하루 10,000, 매일 0시 UTC = 한국 오전 9시 초기화)
#  - 토큰에 'Account Analytics 읽기' 권한이 있으면 Cloudflare 실제 사용량(GraphQL totalNeurons, 1분마다)
#  - 없으면 이 서버가 답할 때마다 받은 토큰 수로 직접 셈(서버가 다시 켜지면 0부터, 사이트 밖 사용량은 빠짐)
# ---------------------------------------------------------
DAILY_NEURONS = 10000
# Gemma 4 26B 단가(1M 토큰당 $0.10 입력·$0.30 출력)를 뉴런으로(1,000뉴런 = $0.011)
NEURONS_PER_TOKEN = (0.10 / 0.011 * 1000 / 1e6, 0.30 / 0.011 * 1000 / 1e6)
_usage = {"day": "", "self": 0.0, "cf": None, "cf_at": 0.0, "cf_ok": None}
_usage_lock = threading.Lock()


def _utc_day() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _add_usage(u: dict) -> None:
    n = u.get("prompt_tokens", 0) * NEURONS_PER_TOKEN[0] + u.get("completion_tokens", 0) * NEURONS_PER_TOKEN[1]
    with _usage_lock:
        if _usage["day"] != _utc_day():
            _usage.update(day=_utc_day(), self=0.0, cf=None)
        _usage["self"] += n


def _fetch_cf_usage():
    """Cloudflare가 집계한 오늘 뉴런 합계. 권한이 없거나 실패하면 None."""
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    q = ("query($a:String!,$s:Time!,$e:Time!){viewer{accounts(filter:{accountTag:$a}){"
         "aiInferenceAdaptiveGroups(limit:100,filter:{datetime_geq:$s,datetime_leq:$e}){sum{totalNeurons}}}}}")
    var = {"a": CF_ACCOUNT_ID, "s": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "e": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    req = urllib.request.Request("https://api.cloudflare.com/client/v4/graphql", data=json.dumps({"query": q, "variables": var}).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + CF_API_TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read().decode("utf-8"))
        accts = ((d.get("data") or {}).get("viewer") or {}).get("accounts") or []
        if d.get("errors") or not accts:
            return None
        return sum(g["sum"]["totalNeurons"] for g in accts[0]["aiInferenceAdaptiveGroups"])
    except Exception:  # noqa: BLE001
        return None


def _refresh_usage_loop():
    while True:
        v = _fetch_cf_usage()
        with _usage_lock:
            if _usage["day"] != _utc_day():
                _usage.update(day=_utc_day(), self=0.0, cf=None)
            if v is not None:
                _usage.update(cf=v, cf_at=time.time())
            if _usage["cf_ok"] is None or _usage["cf_ok"] != (v is not None):
                log("오늘 사용량: " + ("Cloudflare 실제 집계 사용" if v is not None else "권한 없음 → 서버가 직접 셈"))
            _usage["cf_ok"] = v is not None
        time.sleep(60 if v is not None else 600)


def usage() -> dict:
    """페이지 표시용: 오늘 쓴 뉴런·남은 뉴런·출처·초기화 시각."""
    with _usage_lock:
        if _usage["day"] != _utc_day():
            _usage.update(day=_utc_day(), self=0.0, cf=None)
        used = _usage["cf"] if _usage["cf"] is not None else _usage["self"]
        src = "cloudflare" if _usage["cf"] is not None else "server"
    used = round(min(max(used, 0.0), DAILY_NEURONS * 10), 1)
    return {"used": used, "limit": DAILY_NEURONS, "left": round(max(0.0, DAILY_NEURONS - used)), "source": src, "reset_kst": "09:00"}


class Busy(Exception):
    """무료 사용량이 다 찼거나 Cloudflare가 잠시 못 받음 → app.py가 429(잠시 후 다시)로 돌려줌."""


def ready() -> bool:
    return bool(CF_ACCOUNT_ID and CF_API_TOKEN)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [gemma] {msg}", flush=True)


def clean(text: str) -> str:
    return THINK.sub("", text).replace("<think>", "").replace("</think>", "").strip()


def wiki_context(text: str):
    """사실 질문이면 위키백과 요약 [(제목, 요약, 주소), ...] 최대 2건 — 아니면 None."""
    m = FACT_Q.search(text)
    if not m or SELF_Q.search(text):  # "너는 누구야?" 같은 말은 백과사전에서 찾지 않는다
        return None
    rest = re.sub(r"[^가-힣A-Za-z0-9]", "", FACT_Q.sub("", text))
    rest = re.sub(r"(야|예요|이야|니|지|요|냐|어|에요|인가요|일까)$", "", rest)
    if len(rest) < 2:  # "누구야", "언제?"처럼 알맹이가 없으면 찾지 않는다
        return None
    subject = re.sub(r"(은|는|이|가|의|을|를|에서|에는|에|도)$", "", text[:m.start()].strip(" ,?."))
    found, seen = [], set()
    for q in [q for q in (subject, text) if q]:  # 대상 검색 1건 + 문장 통째 검색 1건
        try:
            for title in wiki_search._search(q, 2):
                if title in seen:
                    continue
                seen.add(title)
                s = wiki_search._summary(title)
                if s:
                    found.append(s)
                    break
        except Exception as e:  # noqa: BLE001
            log(f"위키 검색 실패: {type(e).__name__}")
    return found or None


def _call(messages: list) -> dict:
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    body = json.dumps({"model": CF_MODEL, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + CF_API_TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            d = json.loads(r.read().decode("utf-8"))
        _add_usage(d.get("usage") or {})
        return d
    except urllib.error.HTTPError as e:
        log(f"Cloudflare 오류 {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")  # 본문에 열쇠는 없음
        raise Busy() from e
    except Exception as e:  # noqa: BLE001 — 연결 실패·시간 초과
        log(f"Cloudflare 연결 실패: {type(e).__name__}")
        raise Busy() from e


def reply(text: str, history: list):
    """(답, 위키 근거 주소 또는 None). history는 app.py의 Turn 목록(role, content)."""
    ctx = wiki_context(text) if USE_WIKI_CONTEXT else None
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-HISTORY_TURNS:]:
        msgs.append({"role": h.role, "content": h.content[:600]})
    if ctx:
        # 괄호 안(생몰년·높이 같은 숫자)을 지우면 답이 사라진다 → wiki_search._clean은 쓰지 않고 공백만 정리
        # (2026-11-08 사용자: "위키백과 돌려서 오히려 멍청하다") → 자료는 참고만, 관련 없으면 무시하고 아는 대로. 숫자·날짜만 조심.
        docs = "\n\n".join(f"[참고 자료: 위키백과 「{t}」]\n{re.sub(' +', ' ', e.replace(chr(160), ' '))[:800]}" for t, e, _ in ctx)
        msgs.append({"role": "user", "content": f"{docs}\n\n위 자료는 참고용이야. 질문과 관련 있으면 활용하고, 관련 없거나 답이 없으면 자료 이야기는 하지 말고 "
                                                f"네가 아는 대로 자연스럽게 답해. 다만 숫자·날짜는 자료에 있거나 확실히 알 때만 말하고, 확실하지 않으면 정확히는 모른다고 해.\n"
                                                f"질문: {text}"})
    else:
        msgs.append({"role": "user", "content": text})
    d = _call(msgs)
    out = clean(d["choices"][0]["message"]["content"] or "")
    if STRAY.search(out):  # 외국 글자가 섞이면 한 번 더(그래도 섞이면 그 글자만 지움)
        out = STRAY.sub("", clean(_call(msgs)["choices"][0]["message"]["content"] or "")).strip()
    return (out or "..."), (ctx[0][2] if ctx else None)


if ready():
    threading.Thread(target=_refresh_usage_loop, daemon=True).start()
