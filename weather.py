"""날씨 도구 (2026-11-09 사용자: "위치 권한으로 위치를 알 수 있게 할 수 없을까?" — 웹 검색으로 날씨를 찾으면 맛집·PDF만 걸렸음)

- Gemma 4가 <<weather: 지역>>(또는 <<weather: 내 위치>>)를 내면 cf_chat이 부른다.
- 지명 → 좌표: OpenStreetMap Nominatim(열쇠 없음). 한국 지명에 강함(서초구·이수역·제주 OK). Open-Meteo 지명 찾기는 '서울·제주'도 못 찾아서 안 씀.
  Nominatim 규칙(1초에 1번, 앱 이름이 든 User-Agent)을 지키려고 잠금·캐시를 둔다.
- 좌표 → 동네 이름: Nominatim reverse (페이지의 '위치' 단추로 받은 대략 좌표, 소수 둘째 자리 ≈ 1km).
- 예보: Open-Meteo(열쇠 없음, 비상업 무료). 지금 날씨 + 오늘부터 4일.
- 로그에는 지명·좌표를 남기지 않는다.
"""
import datetime
import json
import threading
import time
import urllib.parse
import urllib.request

UA = "QueryBot-AI/1.0 (https://querybot.kr/ai/)"
WMO = {0: "맑음", 1: "대체로 맑음", 2: "구름 조금", 3: "흐림", 45: "안개", 48: "서리 안개", 51: "약한 이슬비", 53: "이슬비", 55: "강한 이슬비",
       56: "약한 어는 이슬비", 57: "어는 이슬비", 61: "약한 비", 63: "비", 65: "강한 비", 66: "약한 어는 비", 67: "어는 비", 71: "약한 눈", 73: "눈",
       75: "강한 눈", 77: "싸락눈", 80: "약한 소나기", 81: "소나기", 82: "강한 소나기", 85: "약한 눈보라", 86: "강한 눈보라", 95: "뇌우",
       96: "우박을 동반한 뇌우", 99: "강한 우박 뇌우"}
WEEK = "월화수목금토일"
KST = datetime.timezone(datetime.timedelta(hours=9))
_cache: dict = {}
_lock = threading.Lock()
_osm_lock = threading.Lock()
_osm_last = [0.0]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [weather] {msg}", flush=True)


def _get(url: str, timeout: int = 8):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ko"}), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _cached(key, ttl, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
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


def geocode(place: str):
    """지명 → (위도, 경도, 이름). 못 찾으면 None."""
    place = " ".join((place or "").split())[:60]
    if not place:
        return None

    def find():
        d = _osm("search", {"q": place, "format": "jsonv2", "limit": 1, "addressdetails": 1, "accept-language": "ko"})
        if not d:
            return None
        x = d[0]
        nm = x.get("name") or ""
        addr = x.get("address") or {}
        lab = _label(addr, nm if nm not in (addr.get("province"), addr.get("city"), addr.get("borough"), addr.get("state")) else "")
        return float(x["lat"]), float(x["lon"]), lab or nm or place
    return _cached(("geo", place), 86400, find)


def reverse(lat: float, lon: float) -> str:
    """대략 좌표 → '서울특별시 동작구 사당2동' 같은 이름. 실패하면 빈 글."""
    lat, lon = round(lat, 2), round(lon, 2)

    def find():
        try:
            d = _osm("reverse", {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14, "addressdetails": 1, "accept-language": "ko"})
            return _label(d.get("address") or {})
        except Exception as e:  # noqa: BLE001
            log(f"동네 이름 찾기 실패: {type(e).__name__}")
            return ""
    return _cached(("rev", lat, lon), 86400, find)


def forecast(lat: float, lon: float) -> dict:
    lat, lon = round(lat, 2), round(lon, 2)
    q = {"latitude": lat, "longitude": lon, "timezone": "Asia/Seoul", "forecast_days": 4,
         "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
         "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum"}
    return _cached(("fc", lat, lon), 900, lambda: _get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q)))


def report(place: str = "", lat=None, lon=None):
    """(모델에게 줄 글, 출처 {title,url}) — 지역을 못 찾거나 예보가 실패하면 (안내 글, None)."""
    t0 = time.time()
    label = ""
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
    cur, day = w.get("current") or {}, w.get("daily") or {}
    today = datetime.datetime.now(KST).date()
    lines = [f"[날씨 도구 결과] 위치: {label} (위도 {round(lat, 2)}, 경도 {round(lon, 2)}), 자료: Open-Meteo 예보, 한국 시각 기준"]
    if cur:
        lines.append(f"지금({str(cur.get('time', ''))[-5:]}): {WMO.get(cur.get('weather_code'), '알 수 없음')}, 기온 {cur.get('temperature_2m')}°C "
                     f"(체감 {cur.get('apparent_temperature')}°C), 습도 {cur.get('relative_humidity_2m')}%, 바람 {cur.get('wind_speed_10m')}km/h, "
                     f"강수 {cur.get('precipitation')}mm")
    for i, ds in enumerate(day.get("time") or []):
        try:
            dd = datetime.date.fromisoformat(ds)
        except ValueError:
            continue
        rel = {0: "오늘", 1: "내일", 2: "모레"}.get((dd - today).days, "")
        lines.append(f"{dd.month}월 {dd.day}일({WEEK[dd.weekday()]}{', ' + rel if rel else ''}): {WMO.get((day.get('weather_code') or [None] * 9)[i], '알 수 없음')}, "
                     f"최저 {day['temperature_2m_min'][i]}°C / 최고 {day['temperature_2m_max'][i]}°C, "
                     f"강수확률 {day['precipitation_probability_max'][i]}%, 강수량 {day['precipitation_sum'][i]}mm")
    log(f"예보 {len(day.get('time') or [])}일 ({time.time() - t0:.1f}초)")
    note = "\n".join(lines) + ("\n\n위 날씨 자료로 원래 질문에 자연스럽게 답해. 어느 지역·어느 날 기준인지 함께 말하고, 숫자는 자료 그대로 써. "
                               "사용자가 이 자료에 없는 것(미세먼지 등)을 물었을 때만 지어내지 말고 이 자료에는 없다고 말해(묻지 않았으면 꺼내지 마). "
                               "출처 번호는 붙이지 마. <<weather>>는 다시 쓰지 마.")
    return note, {"title": f"Open-Meteo 날씨 예보 · {label}", "url": "https://open-meteo.com/"}
