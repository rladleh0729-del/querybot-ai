"""날씨 도구 (2026-11-09 사용자: "위치 권한으로 위치를 알 수 있게 할 수 없을까?" — 웹 검색으로 날씨를 찾으면 맛집·PDF만 걸렸음)

- Gemma 4가 <<weather: 지역>>(또는 <<weather: 내 위치>>)를 내면 cf_chat이 부른다.
- 지명 → 좌표: OpenStreetMap Nominatim(열쇠 없음, 한국 지명에 강함: 서초구·이수역·제주 OK) → 실패하면 카카오 지도 검색(KAKAO_REST_API_KEY).
  Open-Meteo 지명 찾기는 '서울·제주'도 못 찾아서 안 씀. Nominatim 규칙(1초에 1번, 앱 이름이 든 User-Agent)을 지키려고 잠금·캐시를 둔다.
- 좌표 → 동네 이름: Nominatim reverse (페이지의 '위치' 단추로 받은 대략 좌표, 소수 둘째 자리 ≈ 1km).
- 예보: Open-Meteo(열쇠 없음) → 막히면 노르웨이 기상청 MET Norway(열쇠 없음, 앱 이름 User-Agent 필요).
  Render(여러 사이트가 같은 인터넷 주소를 씀)에서는 Open-Meteo가 429(요청 너무 많음)로 막혀서(2026-11-09) MET Norway가 실제로 쓰임.
- 로그에는 지명·좌표를 남기지 않는다.
"""
import datetime
import json
import re
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "QueryBot-AI/1.0 https://querybot.kr/ai/"
KAKAO_KEY = os.environ.get("KAKAO_REST_API_KEY", "").strip()
WMO = {0: "맑음", 1: "대체로 맑음", 2: "구름 조금", 3: "흐림", 45: "안개", 48: "서리 안개", 51: "약한 이슬비", 53: "이슬비", 55: "강한 이슬비",
       56: "약한 어는 이슬비", 57: "어는 이슬비", 61: "약한 비", 63: "비", 65: "강한 비", 66: "약한 어는 비", 67: "어는 비", 71: "약한 눈", 73: "눈",
       75: "강한 눈", 77: "싸락눈", 80: "약한 소나기", 81: "소나기", 82: "강한 소나기", 85: "약한 눈보라", 86: "강한 눈보라", 95: "뇌우",
       96: "우박을 동반한 뇌우", 99: "강한 우박 뇌우"}
# MET Norway 날씨 기호 → 한국어 (뒤의 _day/_night는 떼고 봄)
MET = {"clearsky": "맑음", "fair": "대체로 맑음", "partlycloudy": "구름 조금", "cloudy": "흐림", "fog": "안개",
       "lightrain": "약한 비", "rain": "비", "heavyrain": "강한 비", "lightrainshowers": "약한 소나기", "rainshowers": "소나기",
       "heavyrainshowers": "강한 소나기", "lightsleet": "약한 진눈깨비", "sleet": "진눈깨비", "heavysleet": "강한 진눈깨비",
       "lightsleetshowers": "약한 진눈깨비", "sleetshowers": "진눈깨비", "heavysleetshowers": "강한 진눈깨비", "lightsnow": "약한 눈",
       "snow": "눈", "heavysnow": "강한 눈", "lightsnowshowers": "약한 눈", "snowshowers": "눈", "heavysnowshowers": "강한 눈"}
MET_RANK = ["clearsky", "fair", "partlycloudy", "cloudy", "fog"]  # 그 뒤(비·눈 등)는 더 '센' 날씨
WEEK = "월화수목금토일"
KST = datetime.timezone(datetime.timedelta(hours=9))
_cache: dict = {}
_lock = threading.Lock()
_osm_lock = threading.Lock()
_osm_last = [0.0]
_om_block = [0.0]  # Open-Meteo가 막힌 시각 (30분 동안은 바로 MET Norway로)
# 주요 도시는 좌표를 미리 (2026-11-10: Render에서 Nominatim이 '서울'을 못 찾고 그 실패가 하루 동안 저장돼 '어느 지역?'만 되물음)
KNOWN = {
    "서울": (37.5665, 126.9780, "서울특별시"), "부산": (35.1796, 129.0756, "부산광역시"), "대구": (35.8714, 128.6014, "대구광역시"),
    "인천": (37.4563, 126.7052, "인천광역시"), "광주": (35.1595, 126.8526, "광주광역시"), "대전": (36.3504, 127.3845, "대전광역시"),
    "울산": (35.5384, 129.3114, "울산광역시"), "세종": (36.4800, 127.2890, "세종특별자치시"), "제주": (33.4996, 126.5312, "제주특별자치도 제주시"),
    "제주도": (33.4996, 126.5312, "제주특별자치도 제주시"), "서귀포": (33.2541, 126.5601, "제주특별자치도 서귀포시"),
    "수원": (37.2636, 127.0286, "경기도 수원시"), "성남": (37.4200, 127.1267, "경기도 성남시"), "고양": (37.6584, 126.8320, "경기도 고양시"),
    "용인": (37.2411, 127.1776, "경기도 용인시"), "창원": (35.2280, 128.6811, "경상남도 창원시"), "청주": (36.6424, 127.4890, "충청북도 청주시"),
    "전주": (35.8242, 127.1480, "전북특별자치도 전주시"), "천안": (36.8151, 127.1139, "충청남도 천안시"), "포항": (36.0190, 129.3435, "경상북도 포항시"),
    "춘천": (37.8813, 127.7298, "강원특별자치도 춘천시"), "강릉": (37.7519, 128.8761, "강원특별자치도 강릉시"), "여수": (34.7604, 127.6622, "전라남도 여수시"),
    "경주": (35.8562, 129.2247, "경상북도 경주시"), "김해": (35.2285, 128.8894, "경상남도 김해시"), "원주": (37.3422, 127.9202, "강원특별자치도 원주시"),
    "목포": (34.8118, 126.3922, "전라남도 목포시"), "안동": (36.5684, 128.7294, "경상북도 안동시"),
}
try:  # PC의 파이썬은 인증서 목록이 모자라 api.met.no 연결이 실패한 일이 있어 certifi가 있으면 그걸 쓴다
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [weather] {msg}", flush=True)


def _get(url: str, timeout: int = 8, headers=None):
    h = {"User-Agent": UA, "Accept-Language": "ko"}
    h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout, context=_SSL) as r:
        return json.loads(r.read().decode("utf-8"))


def _cached(key, ttl, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time() if val is not None else time.time() - ttl + 60, val)  # 실패는 1분만
        if len(_cache) > 500:
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:100]:
                del _cache[k]
    return val


def _osm(path: str, params: dict):
    with _osm_lock:  # Nominatim 사용 규칙: 1초에 1번까지
        wait = 1.05 - (time.time() - _osm_last[0])
        if wait > 0:
            time.sleep(wait)
        _osm_last[0] = time.time()
        return _get("https://nominatim.openstreetmap.org/" + path + "?" + urllib.parse.urlencode(params))


def _label(addr: dict, name: str = "") -> str:
    region = addr.get("province") or addr.get("state") or addr.get("city") or ""
    district = addr.get("borough") or addr.get("county") or addr.get("city_district") or addr.get("town") or ""
    if region and addr.get("city") and addr.get("city") != region and not district:
        district = addr["city"]
    local = addr.get("suburb") or addr.get("village") or addr.get("quarter") or ""
    parts = []
    for p in (region, district, local if not name else "", name):
        if p and p not in parts:
            parts.append(p)
    return " ".join(parts) or addr.get("country", "")


def _kakao_geo(place: str):
    """카카오 지도 검색(장소 이름 → 주소 순). 카카오 앱에서 지도 사용이 꺼져 있으면 실패."""
    hdr = {"Authorization": "KakaoAK " + KAKAO_KEY}
    for kind in ("keyword", "address"):
        d = _get(f"https://dapi.kakao.com/v2/local/search/{kind}.json?" + urllib.parse.urlencode({"query": place, "size": 1}), headers=hdr)
        docs = d.get("documents") or []
        if docs:
            x = docs[0]
            name = x.get("place_name") or ""
            addr = x.get("address_name") or (x.get("address") or {}).get("address_name") or ""
            label = " ".join(addr.split()[:2] + ([name] if name else [])) or place
            return float(x["y"]), float(x["x"]), label
    return None


def geocode(place: str):
    """지명 → (위도, 경도, 이름). 못 찾으면 None."""
    place = " ".join((place or "").split())[:60]
    if not place:
        return None
    k = re.sub(r"(특별자치시|특별자치도|특별시|광역시|시)$", "", place.replace(" ", ""))
    if k in KNOWN:  # 주요 도시는 바로
        return KNOWN[k]

    def find():
        try:
            d = _osm("search", {"q": place, "format": "jsonv2", "limit": 1, "addressdetails": 1, "accept-language": "ko"})
            if d:
                x = d[0]
                nm = x.get("name") or ""
                addr = x.get("address") or {}
                lab = _label(addr, nm if nm not in (addr.get("province"), addr.get("city"), addr.get("borough"), addr.get("state")) else "")
                return float(x["lat"]), float(x["lon"]), lab or nm or place
        except Exception as e:  # noqa: BLE001
            log(f"OpenStreetMap 지명 찾기 실패: {type(e).__name__} {str(e)[:50]}")
        if KAKAO_KEY:
            try:
                return _kakao_geo(place)
            except Exception as e:  # noqa: BLE001
                log(f"카카오 지명 찾기 실패: {type(e).__name__} {str(e)[:50]}")
        return None
    g = _cached(("geo", place), 86400, find)
    if g is None:  # '서울 서초구'를 못 찾으면(지명 서비스 시간 초과 등) 앞의 도시라도 — 되묻기만 하지 않게
        k1 = re.sub(r"(특별자치시|특별자치도|특별시|광역시|시|도)$", "", place.split()[0])
        if k1 in KNOWN:
            lat, lon, lab = KNOWN[k1]
            return lat, lon, lab + " (" + place + " 근처는 못 찾아 도시 전체 기준)"
    return g


def reverse(lat: float, lon: float) -> str:
    """대략 좌표 → '서울특별시 동작구 사당2동' 같은 이름. 실패하면 빈 글."""
    lat, lon = round(lat, 2), round(lon, 2)

    def find():
        try:
            d = _osm("reverse", {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14, "addressdetails": 1, "accept-language": "ko"})
            return _label(d.get("address") or {})
        except Exception as e:  # noqa: BLE001
            log(f"동네 이름 찾기 실패: {type(e).__name__}")
        if KAKAO_KEY:
            try:
                d = _get("https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?" + urllib.parse.urlencode({"x": lon, "y": lat}),
                         headers={"Authorization": "KakaoAK " + KAKAO_KEY})
                docs = [x for x in d.get("documents") or [] if x.get("region_type") == "H"] or d.get("documents") or []
                if docs:
                    return docs[0].get("address_name") or ""
            except Exception as e:  # noqa: BLE001
                log(f"카카오 동네 이름 찾기 실패: {type(e).__name__}")
        return ""
    return _cached(("rev", lat, lon), 86400, find)


# ---------- 예보: 서로 다른 두 곳을 같은 모양 {"src", "current": {...}, "days": [...]}으로 ----------
def _open_meteo(lat: float, lon: float) -> dict:
    q = {"latitude": lat, "longitude": lon, "timezone": "Asia/Seoul", "forecast_days": 4,
         "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
         "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum"}
    w = _get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q))
    c, d = w.get("current") or {}, w.get("daily") or {}
    days = []
    for i, ds in enumerate(d.get("time") or []):
        days.append({"date": ds, "desc": WMO.get(d["weather_code"][i], "알 수 없음"), "tmin": d["temperature_2m_min"][i],
                     "tmax": d["temperature_2m_max"][i], "pop": d["precipitation_probability_max"][i], "psum": d["precipitation_sum"][i]})
    cur = {"time": str(c.get("time", ""))[-5:], "desc": WMO.get(c.get("weather_code"), "알 수 없음"), "temp": c.get("temperature_2m"),
           "feels": c.get("apparent_temperature"), "hum": c.get("relative_humidity_2m"), "wind": c.get("wind_speed_10m"), "prec": c.get("precipitation")}
    return {"src": "open-meteo", "current": cur, "days": days}


def _met_desc(code: str) -> str:
    base = (code or "").split("_")[0]
    thunder = "andthunder" in base
    base = base.replace("andthunder", "")
    return MET.get(base, "알 수 없음") + (" (천둥·번개)" if thunder else "")


def _met_rank(code: str) -> int:
    base = (code or "").split("_")[0]
    if "thunder" in base:
        return 99
    if base in MET_RANK:
        return MET_RANK.index(base)
    return 10 + (2 if base.startswith("heavy") else 0 if base.startswith("light") else 1)


def _met_norway(lat: float, lon: float) -> dict:
    d = _get(f"https://api.met.no/weatherapi/locationforecast/2.0/complete?lat={lat}&lon={lon}", headers={"Accept-Language": "en"})
    ts = d.get("properties", {}).get("timeseries") or []
    if not ts:
        raise RuntimeError("MET Norway 예보 없음")
    first = ts[0]["data"]
    inst = first.get("instant", {}).get("details", {})
    n1 = first.get("next_1_hours") or first.get("next_6_hours") or {}
    t0 = datetime.datetime.fromisoformat(ts[0]["time"].replace("Z", "+00:00")).astimezone(KST)
    cur = {"time": t0.strftime("%H:%M"), "desc": _met_desc(n1.get("summary", {}).get("symbol_code")), "temp": inst.get("air_temperature"),
           "feels": inst.get("apparent_air_temperature"), "hum": round(inst.get("relative_humidity", 0)),
           "wind": round(inst.get("wind_speed", 0) * 3.6, 1), "prec": (n1.get("details") or {}).get("precipitation_amount", 0.0)}
    today = datetime.datetime.now(KST).date()
    days, covered = {}, None
    for x in ts:
        t = datetime.datetime.fromisoformat(x["time"].replace("Z", "+00:00"))
        k = t.astimezone(KST)
        if (k.date() - today).days > 3:
            break
        dd = days.setdefault(k.date().isoformat(), {"temps": [], "psum": 0.0, "pop": None, "code": "", "rank": -1})
        data = x["data"]
        temp = data.get("instant", {}).get("details", {}).get("air_temperature")
        if temp is not None:
            dd["temps"].append(temp)
        for key, hours in (("next_1_hours", 1), ("next_6_hours", 6)):
            blk = data.get(key)
            if not blk:
                continue
            det = blk.get("details") or {}
            if det.get("probability_of_precipitation") is not None:  # 한국 지역은 강수확률을 안 주는 일이 많음 → 없으면 칸을 뺀다(0%로 쓰지 않음)
                dd["pop"] = max(dd["pop"] or 0, det["probability_of_precipitation"])
            if covered is None or t >= covered:  # 강수량은 겹치지 않게 한 번씩만 더한다
                dd["psum"] += det.get("precipitation_amount") or 0.0
                covered = t + datetime.timedelta(hours=hours)
            if 6 <= k.hour <= 20:  # 낮 동안의 가장 '센' 날씨를 그날 날씨로
                code = blk.get("summary", {}).get("symbol_code", "")
                if _met_rank(code) > dd["rank"]:
                    dd["rank"], dd["code"] = _met_rank(code), code
            break
    out = []
    for ds, dd in sorted(days.items()):
        if not dd["temps"]:
            continue
        out.append({"date": ds, "desc": _met_desc(dd["code"]) if dd["code"] else "알 수 없음", "tmin": round(min(dd["temps"]), 1),
                    "tmax": round(max(dd["temps"]), 1), "pop": None if dd["pop"] is None else round(dd["pop"]), "psum": round(dd["psum"], 1)})
    return {"src": "met-norway", "current": cur, "days": out}


def forecast(lat: float, lon: float) -> dict:
    lat, lon = round(lat, 2), round(lon, 2)

    def find():
        if time.time() - _om_block[0] > 1800:
            try:
                return _open_meteo(lat, lon)
            except Exception as e:  # noqa: BLE001
                _om_block[0] = time.time()
                log(f"Open-Meteo 실패 → MET Norway: {type(e).__name__} {str(e)[:50]}")
        return _met_norway(lat, lon)
    return _cached(("fc", lat, lon), 900, find)


SOURCES = {"open-meteo": ("Open-Meteo 날씨 예보", "https://open-meteo.com/"),
           "met-norway": ("노르웨이 기상청(MET Norway) 예보", "https://www.met.no/en")}


def report(place: str = "", lat=None, lon=None):
    """(모델에게 줄 글, 출처 {title,url}) — 지역을 못 찾거나 예보가 실패하면 (안내 글, None)."""
    t0 = time.time()
    try:
        if lat is None or lon is None:
            g = geocode(place)
            if not g:
                log(f"지명 못 찾음 ({len(place)}자)")
                return f"날씨 도구가 '{place}'라는 지역을 찾지 못했어. 지역 이름을 다시 물어봐.", None
            lat, lon, label = g
        else:
            label = reverse(lat, lon) or "사용자 위치"
        w = forecast(lat, lon)
    except Exception as e:  # noqa: BLE001
        log(f"실패: {type(e).__name__} {str(e)[:60]}")
        return "날씨 도구가 지금 응답하지 않아. 날씨는 확인하지 못했다고 솔직히 말하고 숫자를 지어내지 마.", None
    src_title, src_url = SOURCES[w["src"]]
    c = w["current"]
    today = datetime.datetime.now(KST).date()
    lines = [f"[날씨 도구 결과] 위치: {label} (위도 {round(lat, 2)}, 경도 {round(lon, 2)}), 자료: {src_title}, 한국 시각 기준"]
    if c:
        # 비가 0이면 숫자 대신 말로 ("강수량은 0.0mm라고 해요"보다 "비 소식은 없어요"가 자연스러움 — 2026-11-09 사용자)
        now_rain = "지금 비·눈 안 옴" if not c.get("prec") else f"지금 비·눈 {c['prec']}mm/시간"
        lines.append(f"지금({c['time']}): {c['desc']}, 기온 {c['temp']}°C (체감 {c['feels']}°C), 습도 {c['hum']}%, 바람 {c['wind']}km/h, {now_rain}")
    for d in w["days"]:
        dd = datetime.date.fromisoformat(d["date"])
        rel = {0: "오늘", 1: "내일", 2: "모레"}.get((dd - today).days, "")
        pop = "" if d.get("pop") is None else f"강수확률 {d['pop']}%, "
        rain = "비·눈 소식 없음" if (d.get("psum") or 0) < 0.1 and (d.get("pop") or 0) < 30 else f"{pop}강수량 {d['psum']}mm"
        lines.append(f"{dd.month}월 {dd.day}일({WEEK[dd.weekday()]}{', ' + rel if rel else ''}): {d['desc']}, 최저 {d['tmin']}°C / 최고 {d['tmax']}°C, {rain}")
    log(f"{w['src']} 예보 {len(w['days'])}일 ({time.time() - t0:.1f}초)")
    note = "\n".join(lines) + ("\n\n위 날씨 자료로 원래 질문에 사람이 말하듯 자연스럽게 답해. 어느 지역·어느 날 기준인지 함께 말하고, 기온 숫자는 자료 그대로 써. "
                               "'강수량 0.0mm'처럼 0인 숫자를 읽어 주지 말고 '비 소식은 없어요'처럼 말해. 습도·바람·강수량 숫자는 물었거나 의미 있을 때(비가 오거나 바람이 셀 때)만 말해. "
                               "날씨에 맞는 짧은 한마디(겉옷·우산 등)를 붙여도 좋아. "
                               "사용자가 이 자료에 없는 것(미세먼지 등)을 물었을 때만 지어내지 말고 이 자료에는 없다고 말해(묻지 않았으면 꺼내지 마). "
                               "출처 번호는 붙이지 마. <<weather>>는 다시 쓰지 마.")
    return note, {"title": f"{src_title} · {label}", "url": src_url}
