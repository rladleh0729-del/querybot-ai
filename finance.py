"""쿼리봇 기업·주가 도구 (2026-11-24 사용자: "현실의 기업들. 방향, 성장 가능성 같은 거 정리한 오픈 자료 … 주가 조회랑 조언용 주식 데이터" → "DART·전일 주가부터")

- 기업: 금융감독원 DART Open API (환경변수 DART_API_KEY) — 기업 개황, 최근 사업·분기보고서의 '사업의 개요'·'기타 참고사항'(산업·성장성),
  주요 재무(최근 3년 + 최근 분기 누적), 최근 공시 목록
- 주가: 공공데이터포털 금융위원회_주식시세정보 (환경변수 DATA_GO_KR_KEY, 데이터 15094808) — 마지막 거래일 종가·등락·거래량·시가총액,
  1년 최고·최저와 1·3·12개월 변화. 실시간 아님(거래일 다음 날 오후에 갱신)
- 열쇠가 없으면 ready()가 False → cf_chat이 도구를 안내하지 않음(코드를 먼저 올려도 안전)
- 모델은 <<stock: 이름>> / <<company: 이름>>을 쓰고, 여기서 만든 글(note)로 답함. 열쇠·요청 주소는 로그에 안 남김
"""
import html
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
DART = "https://opendart.fss.or.kr/api/"
PRICE_URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
PRICE_SRC = "https://www.data.go.kr/data/15094808/openapi.do"
UA = {"User-Agent": "Mozilla/5.0 (querybot)"}
_lock = threading.Lock()
_corps = {"list": [], "at": 0.0, "err": 0.0}
_cache = {}  # 키 → (시각, 값) : 주가 30분, 회사 6시간
ALIAS = {"삼전": "삼성전자", "하닉": "SK하이닉스", "하이닉스": "SK하이닉스", "현차": "현대자동차", "현대차": "현대자동차",
         "네이버": "NAVER", "엔씨": "엔씨소프트", "포스코": "POSCO홀딩스", "포스코홀딩스": "POSCO홀딩스", "엘지": "LG", "에스케이": "SK",
         "lg엔솔": "LG에너지솔루션", "엘지에너지솔루션": "LG에너지솔루션", "삼바": "삼성바이오로직스", "카뱅": "카카오뱅크", "케이티": "KT",
         "기아차": "기아", "셀트": "셀트리온", "삼성sdi": "삼성SDI", "삼성에스디아이": "삼성SDI"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [finance] {msg}", flush=True)


def _key(name: str) -> str:
    v = os.environ.get(name, "").strip()
    return urllib.parse.unquote(v) if "%" in v else v  # 공공데이터포털 '인코딩 키'를 넣어도 되게


def dart_ready() -> bool:
    return bool(_key("DART_API_KEY"))


def price_ready() -> bool:
    return bool(_key("DATA_GO_KR_KEY"))


def ready() -> bool:
    return dart_ready() or price_ready()


def _get(url: str, timeout: int = 20) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _dart(path: str, **params) -> dict:
    params["crtfc_key"] = _key("DART_API_KEY")
    j = json.loads(_get(DART + path + "?" + urllib.parse.urlencode(params)).decode("utf-8"))
    st = j.get("status")
    if st not in ("000", "013"):  # 013 = 조회된 데이터 없음
        raise RuntimeError(f"DART {st} {j.get('message', '')[:60]}")
    return j


def _cached(key: str, ttl: int, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    if len(_cache) > 300:
        _cache.clear()
    _cache[key] = (now, val)
    return val


# ---------- 회사 찾기 ----------
def _norm(s: str) -> str:
    s = (s or "").lower().replace("(주)", "").replace("㈜", "").replace("주식회사", "")
    return re.sub(r"[\s·.,()\-]", "", s)


def corps() -> list:
    """DART 회사 목록 중 상장사만 [(정규화 이름, 회사 이름, corp_code, 종목 코드)] — 하루 한 번 받음"""
    if not dart_ready():
        return []
    now = time.time()
    if _corps["list"] and now - _corps["at"] < 86400:
        return _corps["list"]
    with _lock:
        if _corps["list"] and now - _corps["at"] < 86400:
            return _corps["list"]
        if now - _corps["err"] < 600:
            return _corps["list"]
        try:
            raw = _get(DART + "corpCode.xml?crtfc_key=" + urllib.parse.quote(_key("DART_API_KEY")), timeout=60)
            z = zipfile.ZipFile(io.BytesIO(raw))
            out = []
            with z.open(z.namelist()[0]) as f:
                for _, el in ET.iterparse(f):
                    if el.tag != "list":
                        continue
                    sc = (el.findtext("stock_code") or "").strip()
                    if sc:
                        nm = (el.findtext("corp_name") or "").strip()
                        out.append((_norm(nm), nm, el.findtext("corp_code").strip(), sc))
                    el.clear()
            _corps.update(list=out, at=now)
            log(f"DART 상장사 목록 {len(out)}곳")
        except Exception as e:  # noqa: BLE001
            _corps["err"] = now
            log(f"DART 회사 목록 실패: {type(e).__name__} {str(e)[:80]}")
    return _corps["list"]


def find(name: str):
    """이름 → (회사 이름, corp_code, 종목 코드) 또는 None"""
    q = _norm(ALIAS.get(_norm(name), name))
    q = _norm(ALIAS.get(q, q))
    if not q:
        return None
    lst = corps()
    if re.fullmatch(r"\d{6}", q):
        hit = [c for c in lst if c[3] == q]
        return hit[0][1:] if hit else None
    for test in (lambda n: n == q, lambda n: n.startswith(q), lambda n: q in n):
        hit = sorted((c for c in lst if test(c[0])), key=lambda c: len(c[0]))
        if hit:
            return hit[0][1:]
    if price_ready():  # 종목 이름(현대차·NAVER처럼 회사 이름과 다른 것)으로 한 번 더
        rows = _price_rows(like_name=name.strip(), days=10)
        if rows:
            code = rows[0]["srtnCd"][-6:]
            hit = [c for c in lst if c[3] == code]
            return hit[0][1:] if hit else (rows[0]["itmsNm"], None, code)
    return None


# ---------- 주가 ----------
def _price_rows(code: str = "", like_name: str = "", days: int = 400) -> list:
    begin = (datetime.now(KST) - timedelta(days=days)).strftime("%Y%m%d")
    p = {"serviceKey": _key("DATA_GO_KR_KEY"), "resultType": "json", "numOfRows": 400, "pageNo": 1, "beginBasDt": begin}
    if code:
        p["likeSrtnCd"] = code
    else:
        p["likeItmsNm"] = like_name
    raw = _get(PRICE_URL + "?" + urllib.parse.urlencode(p), timeout=25).decode("utf-8", "replace")
    if not raw.lstrip().startswith("{"):  # 열쇠 오류 등은 XML로 옴
        m = re.search(r"<returnAuthMsg>(.*?)</returnAuthMsg>|<resultMsg>(.*?)</resultMsg>", raw)
        raise RuntimeError("주가 API " + ((m.group(1) or m.group(2)) if m else raw[:60]))
    body = json.loads(raw).get("response", {}).get("body", {})
    items = (body.get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    if like_name and items:  # 이름이 비슷한 여러 종목 → 정확히 같은 이름, 아니면 시가총액이 가장 큰 종목
        last = max(it["basDt"] for it in items)
        today = [it for it in items if it["basDt"] == last]
        exact = [it for it in today if _norm(it["itmsNm"]) == _norm(like_name)]
        pick = exact[0] if exact else max(today, key=lambda it: int(it.get("mrktTotAmt") or 0))
        items = [it for it in items if it["srtnCd"] == pick["srtnCd"]]
    return sorted(items, key=lambda it: it["basDt"], reverse=True)


def _won(v) -> str:
    try:
        n = int(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return str(v)
    sign, a = ("-" if n < 0 else ""), abs(n)
    if a >= 10 ** 12:
        return f"{sign}{a / 10 ** 12:,.1f}조 원"
    if a >= 10 ** 8:
        return f"{sign}{a / 10 ** 8:,.0f}억 원"
    return f"{sign}{a:,}원"


def _day(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d


def price(name: str):
    """(글, 출처 목록) — 마지막 거래일 시세와 1년 흐름"""
    c = find(name)
    code = c[2] if c else ""
    def go():
        return _price_rows(code=code) if code else _price_rows(like_name=name.strip())
    rows = _cached("p:" + (code or _norm(name)), 1800, go)
    if not rows:
        return f"'{name}' 종목의 시세를 찾지 못했어(한국 상장 주식만 있음).", []
    t = rows[0]

    def chg(days):
        target = (datetime.strptime(t["basDt"], "%Y%m%d") - timedelta(days=days)).strftime("%Y%m%d")
        old = next((r for r in rows if r["basDt"] <= target), None)
        if not old:
            return ""
        a, b = int(t["clpr"]), int(old["clpr"])
        return f"{(a - b) / b * 100:+.1f}% ({_day(old['basDt'])} {b:,}원)" if b else ""
    yr = [r for r in rows if r["basDt"] >= (datetime.strptime(t["basDt"], "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")]
    hi = max(yr, key=lambda r: int(r["hipr"]))
    lo = min(yr, key=lambda r: int(r["lopr"]))
    s = (f"[주가 · {t['itmsNm']}({t['srtnCd'][-6:]}, {t.get('mrktCtg', '')}) · 기준일 {_day(t['basDt'])} 종가 — 실시간 아님]\n"
         f"종가 {int(t['clpr']):,}원, 전일 대비 {int(t['vs']):+,}원({float(t['fltRt']):+.2f}%), 시가 {int(t['mkp']):,} · 고가 {int(t['hipr']):,} · 저가 {int(t['lopr']):,}원\n"
         f"거래량 {int(t['trqu']):,}주, 거래대금 {_won(t['trPrc'])}, 시가총액 {_won(t['mrktTotAmt'])}, 상장주식수 {int(t['lstgStCnt']):,}주\n"
         f"1년 최고 {int(hi['hipr']):,}원({_day(hi['basDt'])}), 1년 최저 {int(lo['lopr']):,}원({_day(lo['basDt'])})\n"
         f"변화: 1개월 {chg(30) or '자료 없음'}, 3개월 {chg(91) or '자료 없음'}, 1년 {chg(365) or '자료 없음'}")
    return s, [{"title": f"{t['itmsNm']} 주가 {_day(t['basDt'])} (금융위원회·공공데이터포털)", "url": PRICE_SRC}]


# ---------- DART 회사 정보 ----------
REPRT = {"03": "11013", "06": "11012", "09": "11014", "12": "11011"}
ACCTS = [("매출액", r"^(매출액|수익\(매출액\)|영업수익|매출)$"), ("영업이익", r"^영업이익(\(손실\))?$"),
         ("당기순이익", r"^당기순이익(\(손실\))?$"), ("자산총계", r"^자산총계$"), ("부채총계", r"^부채총계$"), ("자본총계", r"^자본총계$")]


def _fin(corp_code: str, year: str, reprt: str) -> dict:
    j = _dart("fnlttSinglAcnt.json", corp_code=corp_code, bsns_year=year, reprt_code=reprt)
    rows = j.get("list") or []
    fs = "CFS" if any(r.get("fs_div") == "CFS" for r in rows) else "OFS"
    out = {"_fs": "연결" if fs == "CFS" else "별도"}
    for r in rows:
        if r.get("fs_div") != fs:
            continue
        for label, pat in ACCTS:
            if label not in out and re.match(pat, (r.get("account_nm") or "").replace(" ", "")):
                out[label] = r
    return out


def _amt(r, k):
    v = (r or {}).get(k) or ""
    v = v.replace(",", "").strip()
    return int(v) if re.fullmatch(r"-?\d+", v) else None


def _section(xml: str, title_pat: str, cap: int) -> str:
    m = re.search(r"<TITLE[^>]*>\s*[\dⅠⅡⅢⅣⅤⅥⅦ.\s]*" + title_pat + r"[^<]*</TITLE>", xml)
    if not m:
        return ""
    end = xml.find("<TITLE", m.end())
    chunk = xml[m.end(): end if end > 0 else m.end() + 200000][:200000]
    chunk = re.sub(r"<(TABLE|TABLE-GROUP)\b.*?</\1>", " (표 생략) ", chunk, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", chunk))
    t = re.sub(r"[ \t　\xa0]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t).strip()
    t = re.sub(r"(\s*\(표 생략\)\s*)+", " (표 생략) ", t)
    return (t[:cap] + "…") if len(t) > cap else t


def _overview(rcept_no: str) -> dict:
    raw = _get(DART + "document.xml?" + urllib.parse.urlencode({"crtfc_key": _key("DART_API_KEY"), "rcept_no": rcept_no}), timeout=60)
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = sorted(z.namelist(), key=lambda n: (not n.startswith(rcept_no), -z.getinfo(n).file_size))
    xml = z.read(names[0]).decode("utf-8", "replace")
    del raw, z
    return {"overview": _section(xml, r"사업의\s*개요", 2200), "etc": _section(xml, r"기타\s*참고\s*사항", 1300)}


def company(name: str):
    """(글, 출처 목록) — DART 개황·사업 개요·재무·최근 공시 (+ 주가 열쇠가 있으면 주가)"""
    c = find(name)
    if not c or not c[1]:
        return f"'{name}'에 맞는 한국 상장사를 DART에서 찾지 못했어(비상장·해외 기업은 없음). 회사 이름을 정확히 확인해 달라고 하거나 <<search>>로 찾아.", []
    corp_name, cc, code = c

    def go():
        info = _dart("company.json", corp_code=cc)
        today = datetime.now(KST)
        reps = _dart("list.json", corp_code=cc, bgn_de=(today - timedelta(days=500)).strftime("%Y%m%d"),
                     end_de=today.strftime("%Y%m%d"), pblntf_ty="A", page_count=20).get("list") or []
        reps = [r for r in reps if re.search(r"(사업|반기|분기)보고서", r.get("report_nm", "")) and "기재정정" not in r.get("rm", "")]
        recent = _dart("list.json", corp_code=cc, bgn_de=(today - timedelta(days=90)).strftime("%Y%m%d"),
                       end_de=today.strftime("%Y%m%d"), page_count=8).get("list") or []
        annual = next((r for r in reps if "사업보고서" in r["report_nm"]), None)
        latest = reps[0] if reps else None
        fin_a = fin_q = None
        if annual:
            y = re.search(r"\((\d{4})\.(\d{2})\)", annual["report_nm"])
            if y:
                fin_a = (y.group(1), _fin(cc, y.group(1), "11011"))
        if latest and latest is not annual:
            y = re.search(r"\((\d{4})\.(\d{2})\)", latest["report_nm"])
            if y and y.group(2) in REPRT:
                fin_q = (f"{y.group(1)}.{y.group(2)}", _fin(cc, y.group(1), REPRT[y.group(2)]))
        ov = {}
        if latest:
            try:
                ov = _overview(latest["rcept_no"])
            except Exception as e:  # noqa: BLE001 — 본문을 못 받아도 나머지는 줌
                log(f"사업 개요 못 받음: {type(e).__name__}")
        return info, annual, latest, recent, fin_a, fin_q, ov

    info, annual, latest, recent, fin_a, fin_q, ov = _cached("c:" + cc, 6 * 3600, go)
    cls = {"Y": "코스피", "K": "코스닥", "N": "코넥스"}.get(info.get("corp_cls"), "기타")
    lines = [f"[기업 · {info.get('corp_name', corp_name)}({info.get('stock_name', '')}, {code}, {cls}) — 금융감독원 DART 공시]",
             f"대표 {info.get('ceo_nm', '')}, 설립 {_day(info.get('est_dt', ''))}, 결산 {info.get('acc_mt', '')}월, 업종 코드 {info.get('induty_code', '')}, "
             f"누리집 {info.get('hm_url', '') or '없음'}"]
    if fin_a:
        y, f = fin_a
        cols = [("thstrm_amount", int(y)), ("frmtrm_amount", int(y) - 1), ("bfefrmtrm_amount", int(y) - 2)]
        lines.append(f"\n주요 재무({f['_fs']}, 사업보고서 기준, 연도별):")
        for label, _ in ACCTS:
            r = f.get(label)
            if r:
                vals = [f"{yy}년 {_won(_amt(r, k))}" for k, yy in cols if _amt(r, k) is not None]
                lines.append(f"- {label}: " + ", ".join(reversed(vals)))
        sales, op, debt, eq = (_amt(f.get(k), "thstrm_amount") for k in ("매출액", "영업이익", "부채총계", "자본총계"))
        extra = []
        if sales and op is not None:
            extra.append(f"영업이익률 {op / sales * 100:.1f}%")
        if debt is not None and eq:
            extra.append(f"부채비율 {debt / eq * 100:.0f}%")
        if extra:
            lines.append(f"- {y}년 " + ", ".join(extra))
    if fin_q:
        p, f = fin_q
        lines.append(f"\n최근 보고서({p}, {f['_fs']}) 누적 실적과 전년 같은 기간:")
        for label in ("매출액", "영업이익", "당기순이익"):
            r = f.get(label)
            if r:
                now_v = _amt(r, "thstrm_add_amount") if _amt(r, "thstrm_add_amount") is not None else _amt(r, "thstrm_amount")
                old_v = _amt(r, "frmtrm_add_amount") if _amt(r, "frmtrm_add_amount") is not None else _amt(r, "frmtrm_amount")
                g = f" ({(now_v - old_v) / abs(old_v) * 100:+.0f}%)" if now_v is not None and old_v else ""
                lines.append(f"- {label}: {_won(now_v)} (전년 같은 기간 {_won(old_v)}){g}")
    if ov.get("overview"):
        lines.append(f"\n사업의 개요({latest['report_nm']} 원문 일부):\n{ov['overview']}")
    if ov.get("etc"):
        lines.append(f"\n기타 참고사항(산업·시장 여건 등, 원문 일부):\n{ov['etc']}")
    if recent:
        lines.append("\n최근 90일 공시: " + "; ".join(f"{_day(r['rcept_dt'])} {r['report_nm'].strip()}" for r in recent[:8]))
    src = []
    if latest:
        src.append({"title": f"{corp_name} {latest['report_nm'].strip()} (DART)", "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={latest['rcept_no']}"})
    if annual and annual is not latest:
        src.append({"title": f"{corp_name} {annual['report_nm'].strip()} (DART)", "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={annual['rcept_no']}"})
    text = "\n".join(lines)
    if price_ready():
        try:
            ps, psrc = price(code or corp_name)
            text += "\n\n" + ps
            src += psrc
            if fin_a:  # 어림 지표: 시가총액 ÷ 연간 순이익·자본
                m = re.search(r"시가총액 ([\d,.]+)(조|억) 원", ps)
                f = fin_a[1]
                ni, eq = _amt(f.get("당기순이익"), "thstrm_amount"), _amt(f.get("자본총계"), "thstrm_amount")
                if m:
                    cap = float(m.group(1).replace(",", "")) * (10 ** 12 if m.group(2) == "조" else 10 ** 8)
                    est = []
                    if ni and ni > 0:
                        est.append(f"PER 약 {cap / ni:.1f}배")
                    if eq and eq > 0:
                        est.append(f"PBR 약 {cap / eq:.2f}배")
                    if est:
                        text += f"\n어림 지표({fin_a[0]}년 연간 실적 기준, 지배주주 몫을 구분하지 않은 대략값): " + ", ".join(est)
        except Exception as e:  # noqa: BLE001
            log(f"주가 붙이기 실패: {type(e).__name__}")
    return text, src


ANSWER_NOTE = (
    "\n\n위 자료는 네가 도구로 직접 찾아본 금융감독원 DART 공시·금융위원회 주가 자료야(사용자가 준 자료처럼 말하지 마). 이걸로 원래 질문에 답해. "
    "숫자는 자료 그대로 쓰고 주가는 기준일을 밝혀(실시간 값이 아님). "
    "회사 방향·성장 가능성을 물으면 공시의 사업 개요·산업 여건과 실적 흐름(매출·이익 증감, 이익률, 부채비율)을 근거로 정리하고, "
    "좋은 점과 위험 요인을 함께 말해. 투자 판단을 물어도 사라·팔라고 단정하지 말고 판단에 쓸 근거와 따져 볼 점을 정리해 준 뒤, "
    "투자 권유가 아니고 판단과 책임은 본인에게 있다는 점을 끝에 한 줄로 밝혀. 자료에 없는 숫자는 지어내지 말고, 최신 뉴스가 필요하면 그렇다고 말해. "
    "<<stock>>·<<company>>는 다시 쓰지 마."
)


def tool(kind: str, arg: str):
    """cf_chat이 부름 → (모델에게 줄 글, 출처 목록)"""
    names = [n.strip() for n in re.split(r"[,，/·]| 와 | 과 | 랑 ", arg) if n.strip()][: (3 if kind == "stock" else 2)]
    parts, src = [], []
    for n in names or [arg]:
        try:
            if kind == "company" and dart_ready():
                t, s = company(n)
            elif price_ready():
                t, s = price(n)
            else:
                t, s = "주가 자료 열쇠가 아직 없어 시세를 못 봤어.", []
        except (urllib.error.URLError, TimeoutError) as e:
            log(f"{kind} 연결 실패: {type(e).__name__}")
            t, s = f"'{n}' 자료 서버에 연결하지 못했어. 지금은 확인할 수 없다고 말해.", []
        except Exception as e:  # noqa: BLE001
            log(f"{kind} 실패: {type(e).__name__} {str(e)[:80]}")
            t, s = f"'{n}' 자료를 가져오다 오류가 났어({type(e).__name__}). 지금은 확인할 수 없다고 말해.", []
        parts.append(t)
        src += s
    log(f"{kind} 도구: {len(names)}곳, 출처 {len(src)}개")
    return "\n\n".join(parts) + ANSWER_NOTE, src
