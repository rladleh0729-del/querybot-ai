"""주인 전용 학교 내신 기출 (2026-11-25 사용자: "내 학교 기출 문제들 다운 받아서 창고에 넣자. 그걸로 기출 내주라 하면 편하겠다"
→ "입체도형이 많으니 이미지는 이미지 그대로").

- 자료: 주인 학교의 수학 정기고사 기출(문항별 원본 그림 + 옮긴 글·보기·정답). Cloudflare D1 표 kb_school에만 두고(공개 GitHub에 없음),
  주인 계정으로 로그인했을 때만 씀 — 학교가 공개 기간을 정해 두고 판매되는 자료라 다른 방문자에게는 안 보임
- want(글) → "1학년 기출 내줘"·"2학년 2학기 기말 기출 하나"처럼 학교 기출을 달라는 말이면 조건(학년·학기·중간/기말·연도)
- pick(조건) → 문항 하나(최근에 낸 것은 피함), quiz(문항) → 모델 없이 바로 낼 답 글(```quiz 카드)과 문항 그림
- ref(앞 답) → 앞에서 낸 문항(풀이를 물으면 그 그림과 공식 정답을 모델에게 같이 줌)
"""
import base64
import collections
import json
import os
import random
import re
import threading
import time
import urllib.request

DB_NAME = os.environ.get("ACCT_D1_NAME", "querybot-chat")
REFRESH = 6 * 3600
_lock = threading.Lock()
_st = {"items": [], "loaded": 0.0, "db": "", "err": ""}
_imgs = collections.OrderedDict()  # id → data URL (최근 40개)
_recent = collections.deque(maxlen=40)  # 최근에 낸 문항 (같은 문항을 또 내지 않게)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [school] {msg}", flush=True)


def _cf(method: str, path: str, body=None):
    acc, tok = os.environ.get("CF_ACCOUNT_ID", "").strip(), os.environ.get("CF_API_TOKEN", "").strip()
    req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{acc}{path}",
                                 data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _q(sql: str, params=None) -> list:
    if not _st["db"]:
        found = _cf("GET", f"/d1/database?name={DB_NAME}")
        dbs = [x for x in found.get("result") or [] if x.get("name") == DB_NAME]
        if not dbs:
            raise RuntimeError("D1 데이터베이스 없음")
        _st["db"] = dbs[0]["uuid"]
    j = _cf("POST", f"/d1/database/{_st['db']}/query", {"sql": sql, "params": params or []})
    return (j.get("result") or [{}])[0].get("results") or []


def items() -> list:
    now = time.time()
    if _st["items"] and now - _st["loaded"] < REFRESH:
        return _st["items"]
    with _lock:
        if _st["items"] and now - _st["loaded"] < REFRESH:
            return _st["items"]
        if _st["err"] and now - _st["loaded"] < 300:
            return _st["items"]
        try:
            rows = _q("SELECT id, school, grade, semester, term, year, no, kind, points, text, choices, answer, figure, solution, full, (fig IS NOT NULL AND fig != '') AS has_fig FROM kb_school")
            for r in rows:
                try:
                    r["choices"] = json.loads(r.get("choices") or "null")
                except ValueError:
                    r["choices"] = None
            _st.update(items=rows, loaded=now, err="")
            log(f"학교 기출 {len(rows)}문항 읽음")
        except Exception as e:  # noqa: BLE001 — 표가 아직 없거나 안 되면 이 기능만 조용히 꺼짐
            _st.update(loaded=now, err=type(e).__name__)
            log(f"학교 기출 읽기 실패: {type(e).__name__}")
    return _st["items"]


def image(item_id: str, col: str = "img") -> str:
    """문항 그림 data URL — col="img" 문항 전체(모델이 풀 때), "fig" 도형·그래프 부분만(사용자에게 보임). 최근 40개는 기억"""
    col = "fig" if col == "fig" else "img"
    ck = col + "|" + item_id
    if ck in _imgs:
        _imgs.move_to_end(ck)
        return _imgs[ck]
    rows = _q(f"SELECT {col} AS b FROM kb_school WHERE id = ?", [item_id])
    b64 = (rows[0].get("b") if rows else "") or ""
    url = "data:image/jpeg;base64," + b64 if b64 else ""
    if url:
        _imgs[ck] = url
        while len(_imgs) > 40:
            _imgs.popitem(last=False)
    return url


def pictures(it) -> list:
    """사용자에게 붙일 그림 (2026-11-26 사용자: '텍스트는 텍스트로, 이미지만 이미지 첨부로')
    보기가 그림이거나 글을 못 뽑은 문항 → 문항 전체 그림(full: 페이지가 보기를 번호만 보여 줌), 도형이 있으면 그 부분만, 글뿐이면 없음"""
    if it.get("full") or not (it.get("text") or "").strip():
        src = image(it["id"], "img")
        return [{"src": src, "alt": label(it), "full": True}] if src else []
    if it.get("has_fig"):
        src = image(it["id"], "fig")
        return [{"src": src, "alt": label(it) + " 그림"}] if src else []
    return []


TEX_FIX = [(r"\overparen", r"\overgroup"), (r"\backsim", r"\sim"), (r"\fallingdotseq", r"\doteq")]  # KaTeX가 못 그리는 명령


def _clean(s: str) -> str:
    s = re.sub(r"^\s*\d{1,3}\s*[.)]\s*", "", (s or "").strip())  # 맨 앞 '13.'·'2)' 같은 번호
    for a, b in TEX_FIX:
        s = s.replace(a, b)  # 글자 그대로 바꿈(정규식 아님)
    return s


ASK = re.compile(r"(기출|시험\s?문제|내신|중간\s?고사|기말\s?고사|중간|기말)")
GIVE = re.compile(r"내\s?줘|내\s?주|내\s?봐|낼래|줘|주세요|줄래|풀래|풀고\s?싶|풀어\s?볼|하나|한\s?문제|문제\s?더|더\s?줘|다음\s?문제|출제|연습")
NOT_SCHOOL = re.compile(r"수능|평가원|모의\s?고사|모의\s?평가|학력\s?평가|대학수학능력")


def want(text: str):
    """학교 기출을 달라는 말이면 조건 dict, 아니면 None"""
    t = text or ""
    if NOT_SCHOOL.search(t) or not ASK.search(t) or not GIVE.search(t) or len(t) > 120:
        return None
    if not re.search(r"학년|학기|중간|기말|내신|우리\s?학교|학교|중\s?[123]|기출", t):
        return None
    c = {}
    m = re.search(r"([123])\s*학년|중\s*([123])(?![\d])", t)
    if m:
        c["grade"] = int(m.group(1) or m.group(2))
    m = re.search(r"([12])\s*학기", t) or re.search(r"[123]\s*-\s*([12])", t)
    if m:
        c["semester"] = int(m.group(1))
    m = re.search(r"중간|기말", t)
    if m:
        c["term"] = m.group(0)
    m = re.search(r"(20\d\d)", t)
    if m:
        c["year"] = int(m.group(1))
    if re.search(r"서술|서답", t):
        c["kind"] = "서술형"
    elif re.search(r"객관식|선택형", t):
        c["kind"] = "객관식"
    if re.search(r"그림|도형|입체", t):
        c["figure"] = True
    return c


MORE = re.compile(r"^\s*(다음\s?(문제|거|꺼)?|하나\s?더|한\s?(문제|개)\s?더|또\s?(줘|내\s?줘|하나|내\s?봐)?|더\s?(줘|내\s?줘|풀래)|계속|next)\s*(줘|요|주세요)?\s*[!.?~]*\s*$", re.I)


def more(text: str, history: list):
    """바로 앞에 학교 기출을 냈고 '하나 더'·'다음 문제'라고 하면 같은 학년·학기 조건"""
    if not MORE.match(text or ""):
        return None
    r = ref(history)
    return {"grade": r["grade"], "semester": r["semester"]} if r else None


def pick(cond: dict):
    its = items()
    if not its:
        return None
    pool = [it for it in its if all(it.get(k) == v for k, v in cond.items() if k not in ("figure", "kind"))]
    if cond.get("kind"):
        pool = [it for it in pool if (it.get("kind") or "") == cond["kind"]] or pool
    if cond.get("figure"):
        pool = [it for it in pool if it.get("figure")] or pool
    fresh = [it for it in pool if it["id"] not in _recent] or pool
    if not fresh:
        return None
    it = random.choice(fresh)
    _recent.append(it["id"])
    return it


def summary() -> str:
    """창고에 있는 시험 목록(조건에 맞는 문항이 없을 때 알려 줌)"""
    exams = sorted({(it["grade"], it["semester"], it["term"], it["year"]) for it in items()})
    by = collections.defaultdict(list)
    for g, s, t, y in exams:
        by[(g, s, t)].append(str(y))
    return ", ".join(f"{g}학년 {s}학기 {t}({'·'.join(ys)})" for (g, s, t), ys in sorted(by.items()))


def label(it) -> str:
    return f"{it['school']} {it['year']}년 {it['grade']}학년 {it['semester']}학기 {it['term']}고사 {it['no'] if it['no'] < 100 else '서술형 ' + str(it['no'] - 100)}번"


CIRCLED = "①②③④⑤⑥⑦⑧⑨"


def _nums(ans: str) -> list:
    out = [CIRCLED.index(ch) + 1 for ch in ans if ch in CIRCLED]
    if not out:
        out = [int(x) for x in re.findall(r"(?<!\d)([1-9])(?!\d)", ans)]
    return out


def quiz(it) -> str:
    """모델 없이 바로 낼 답: 문항 이름(다시 찾는 표시이기도 함) + 퀴즈 카드"""
    head = f"**{label(it)}**" + (f" ({it['points']:g}점)" if isinstance(it.get("points"), (int, float)) and it["points"] else "")
    ch = it.get("choices") or []
    ans = (it.get("answer") or "").strip()
    nums = _nums(ans) if ch else []
    full = bool(it.get("full") or not (it.get("text") or "").strip())
    q = "위 문제를 풀고 답을 골라 보세요." if full else _clean(it["text"])  # 그림이 문항 전체면 안내만, 아니면 문제 글 그대로
    if ch and len(nums) == 1:
        card = {"q": q, "choices": [_clean(re.sub(r"^\s*[①②③④⑤⑥⑦⑧⑨]\s*", "", c)) for c in ch], "answer": nums[0]}
    elif ch and len(nums) > 1:
        a = ", ".join(str(n) for n in nums)
        opts = "" if full else "\n\n" + "\n".join(_clean(c) for c in ch)  # 답이 여럿이면 단답 칸 카드라 보기를 글로 붙임
        card = {"q": q + opts + f"\n\n(정답 {len(nums)}개 — 번호를 쉼표로, 예: 1, 3)", "answer": a,
                "accept": [a.replace(" ", ""), " ".join(str(n) for n in nums), "".join(CIRCLED[n - 1] for n in nums), "".join(str(n) for n in nums)]}
    elif ans:
        bare = re.sub(r"[\s$]", "", ans)  # '$x=3$' → 'x=3', 그리고 '=' 오른쪽 '3'도 맞게
        acc = [bare, bare.split("=")[-1]] if "=" in bare else [bare]
        card = {"q": "위 문제의 답을 써 보세요." if full else q, "answer": ans.replace("$", "") if ans.count("$") == 2 and ans.startswith("$") else ans,
                "accept": list(dict.fromkeys(a for a in acc if a))}
    else:
        return head + ("" if full else "\n\n" + q) + "\n\n다 풀면 답을 보내 주세요(이 문항은 정답표가 없어서 함께 확인해 볼게요)."
    return head + "\n\n```quiz\n" + json.dumps(card, ensure_ascii=False) + "\n```"


REF_CORE = r"(\S+학교) (20\d\d)년 ([123])학년 ([12])학기 (중간|기말)고사 (서술형 )?(\d+)번"
REF = re.compile(r"\*\*" + REF_CORE + r"\*\*")


def find_label(text: str):
    """'이수중학교 2024년 1학년 2학기 기말고사 10번' 같은 이름 → 문항 (저장된 대화를 다시 열 때 그림을 다시 받음)"""
    m = re.search(REF_CORE, text or "")
    return _match(m) if m else None


def _match(m):
    y, g, s, term, sv, no = int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5), m.group(6), int(m.group(7))
    no = no + 100 if sv else no
    for it in items():
        if it["year"] == y and it["grade"] == g and it["semester"] == s and it["term"] == term and it["no"] == no and it["school"] == m.group(1):
            return it
    return None


def ref(history: list):
    """바로 앞 답(들)에서 낸 학교 기출 문항"""
    for h in reversed(history[-4:]):
        role = h.role if hasattr(h, "role") else h.get("role")
        content = h.content if hasattr(h, "content") else h.get("content", "")
        if role != "assistant":
            continue
        m = REF.search(content or "")
        if not m:  # 가장 최근 답이 기출 문제가 아니면 끝(옛 문제를 다시 끌어오지 않음)
            return None
        return _match(m)
    return None


def note(it) -> str:
    ch = it.get("choices") or []
    s = (f"[주인 학교 기출 · {label(it)}] 이 대화에서 사용자에게 낸 문제야. 원본 문항 그림을 같이 줄게(그림이 정확함, 아래 글은 옮긴 것이라 틀릴 수 있음).\n"
         f"문제: {it.get('text') or ''}")
    if ch:
        s += "\n보기: " + "  ".join(ch)
    if it.get("answer"):
        s += f"\n공식 정답: {it['answer']}"
    if it.get("solution"):
        s += f"\n시험지에 딸린 해설(옮긴 글, 참고용): {it['solution'][:1200]}"
    s += ("\n사용자가 답을 말했으면 공식 정답으로 채점하고, 풀이를 물으면 그림을 보고 단계별로 풀어 공식 정답에 닿게 설명해. "
          "아직 답하지 않았는데 다른 걸 물으면 정답을 먼저 말하지 마.")
    return s
