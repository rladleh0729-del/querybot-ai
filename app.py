"""querybot.kr/ai 페이지용 답변 서버 (클라우드판).

PC판(ai_server.py)과 같은 일을 하지만 ngrok 없이 클라우드(Render)에서 24시간 돈다.
포트는 환경변수 PORT를 따르고(Render), 없으면 7860.
PyTorch 대신 numpy로 모델을 돌린다(np_model.py) — Render 무료 메모리 512MB에서 PyTorch는 혼자 350MB를 넘게 먹는다.

모델 파일 (PC에서는 model_files/ 안, GitHub에서는 app.py 옆 — 웹 업로드는 폴더를 못 만들어서):
  model.npz.part01, 02 ...        ← 기본(최신) 모델. make_cloud_model.py로 바꿔서 9MB씩 자른 가중치
  spm_small_chat_gpt.model        ← 기본 모델의 토크나이저
  <이름>.model.npz.part01 ...     ← 방문자가 골라 볼 수 있는 다른 모델 (EXTRA_MODELS)
  <이름>.spm.model                   (make_cloud_model.py --prefix <이름>)

방문자는 모델을 고르고, '이전 대화 기억'을 켜면 앞 대화(history)를 같이 보낸다.
서버는 대화를 저장하지 않는다 — 페이지가 들고 있다가 매번 보낸다.
"""
import gc
import json
import os
import re
import threading
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import sentencepiece as spm
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from np_model import NPModel
from np_sampling import banned_ids, pick_next, space_id_of
import wiki_search  # 지식 질문은 위키백과에서 찾아 답함 (2026-10-20)
import cf_chat  # 'Gemma 4'(Cloudflare Workers AI 무료) — 열쇠(CF_ACCOUNT_ID·CF_API_TOKEN)가 있을 때만 목록에 나옴 (2026-11-08)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("AI_MODEL_DIR", str(
    BASE_DIR / "model_files" if (BASE_DIR / "model_files").is_dir() else BASE_DIR)))

# 고를 수 있는 다른 모델: (키, 이름, 파일 앞머리). 기본 모델은 "latest".
# Render 무료(메모리 512MB)에 맞추려고 처음 고를 때 불러오고, 한 번에 하나만 올려 둔다.
EXTRA_MODELS = [
    ("stage3", "3단계 모델", "stage3"),
    ("v1", "첫 모델 (1,380만)", "v1"),
]

PORT = int(os.environ.get("PORT", "7860"))

MAX_INPUT_CHARS = 200
MAX_HISTORY = 20         # 앞 대화는 최근 20개(10번 주고받기)까지만 받는다
PROMPT_HISTORY = 4       # 그중 모델에 넣는 건 최근 2번 주고받은 것만 (작은 모델은 앞 대화가 길면 헷갈림, 2026-10-21)
# 주인이 사이트 '가르치기 모드'로 가르친 대화(PC 답변 서버에 저장)를 몇 분마다 받아 와서, 같은 말엔 바로 그 답을 한다.
TEACH_PAIRS_URL = os.environ.get("TEACH_PAIRS_URL", "https://poem-defog-septum.ngrok-free.dev/teach/pairs")
TEACH_REFRESH_SEC = 300
MAX_BODY_BYTES = 16384   # 요청 본문 상한 (질문 200자 + 앞 대화 20개)
MAX_CHAT_BODY_BYTES = 6_000_000  # /chat 상한: Gemma 4 그림(페이지가 줄여 보내 장당 보통 0.1~0.5MB)·파일 첨부
MAX_NEW_TOKENS = 80      # 답 자리. 답은 보통 20~40토큰이라 80이면 충분 → 그만큼 앞 대화를 더 넣는다(256 중 176, 2026-10-21)
MAX_IN_FLIGHT = 3        # 답 만드는 중 1개 + 기다리는 2개까지. 넘치면 바로 "잠시 후"
TEMPERATURE = 0.6        # 평가(2026-10-19)에서 0.5~0.6이 0.7보다 조금 나았음
TOP_K = 40
REPETITION = 1.1         # 반복 벌점 (약하게. 세면 조사·어미가 망가짐)
NO_REPEAT_NGRAM = 3      # 이번 답에 나온 3조각 묶음은 다시 못 씀
RATE_LIMIT = 20          # 한 사람(IP)당
RATE_WINDOW_SEC = 60     # 이 시간 안에 최대 RATE_LIMIT번
# 클라우드 앞단 프록시가 X-Forwarded-For 뒤에 덧붙이는 값 개수. 오른쪽에서 이만큼 들어간 값이 진짜 접속 주소.
PROXY_HOPS = max(1, int(os.environ.get("AI_PROXY_HOPS", "1")))
SPECIALS = ("<|document_end|>", "<|user|>", "<|assistant|>", "<|system|>", "<|document_start|>")

ALLOWED_ORIGINS = [
    "https://querybot.kr",
    "https://www.querybot.kr",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
] + [o for o in os.environ.get("AI_EXTRA_ORIGINS", "").split(",") if o]  # 시험할 때만


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================
# 모델 불러오기
# =========================================================
def model_files(prefix: str):
    """(체크포인트 경로, 토크나이저 경로). 기본 모델은 앞머리가 없다."""
    if prefix:
        return MODEL_DIR / f"{prefix}.model.npz", MODEL_DIR / f"{prefix}.spm.model"
    return MODEL_DIR / "model.npz", MODEL_DIR / "spm_small_chat_gpt.model"


def has_files(prefix: str) -> bool:
    ckpt, spm_path = model_files(prefix)
    return (ckpt.exists() or any(MODEL_DIR.glob(ckpt.name + ".part*"))) and spm_path.exists()


class Loaded:
    def __init__(self, prefix: str):
        npz_path, spm_path = model_files(prefix)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(spm_path))
        self.model = NPModel(npz_path)
        meta = self.model.meta
        self.block_size = self.model.block_size
        self.info = {
            "params": self.model.params,
            "iter": int(meta.get("iter_num", 0)),
            "val_loss": round(float(meta.get("best_val_loss", 0.0)), 3),
            "saved_at": meta.get("saved_at", ""),
        }
        unk = self.sp.unk_id()
        self.stops = {i for i in (self.sp.piece_to_id(s) for s in SPECIALS) if i != unk}
        self.space_id = space_id_of(self.sp)  # 첫 토큰 빈칸 막기용 (ai_sampling.py 설명 참고)


if not has_files(""):
    raise SystemExit(f"기본 모델 파일이 없음: {MODEL_DIR}")
loaded = {"latest": Loaded("")}
gc.collect()
log(f"기본 모델 준비: {loaded['latest'].info}")
CHOICES = [("latest", "최신 모델", "")] + [m for m in EXTRA_MODELS if has_files(m[2])]
GEMMA_KEY = "gemma4"
if cf_chat.ready():  # 파일 없이 Cloudflare가 답하는 모델 (앞머리 None)
    CHOICES.append((GEMMA_KEY, "Gemma 4 (구글 26B)", None))
log(f"Gemma 4(Cloudflare): {'켜짐' if cf_chat.ready() else '열쇠 없음 → 목록에서 뺌'}")
load_lock = threading.Lock()


def memory_mb() -> dict:
    """지금·최대 메모리(MB). Render 무료는 메모리 그래프를 안 보여 줘서 /state로 직접 본다 (리눅스에서만)."""
    try:
        status = Path("/proc/self/status").read_text()
        val = lambda key: int(next(l.split()[1] for l in status.splitlines() if l.startswith(key))) // 1024  # noqa: E731
        return {"rss": val("VmRSS:"), "peak": val("VmHWM:")}
    except Exception:  # noqa: BLE001
        return {}


def get_model(key: str) -> Loaded:
    with load_lock:
        if key not in loaded:
            prefix = next((p for k, _, p in CHOICES if k == key), None)
            if prefix is None:
                raise HTTPException(status_code=400, detail="없는 모델")
            # 512MB에 맞추려고 다른 모델은 한 번에 하나만 올려 둔다 (기본 모델은 항상 유지).
            with generate_lock:
                for other in [k for k in loaded if k != "latest"]:
                    del loaded[other]
                gc.collect()
            t0 = time.time()
            loaded[key] = Loaded(prefix)
            gc.collect()
            log(f"모델 불러옴: {key} ({time.time() - t0:.1f}초) {loaded[key].info} 메모리 {memory_mb()}")
        return loaded[key]


generate_lock = threading.Lock()
in_flight = threading.BoundedSemaphore(MAX_IN_FLIGHT)
# Gemma 4는 Cloudflare가 계산해서 이 서버 CPU를 거의 안 쓴다. '깊게 생각'은 30초 넘게 걸리므로 기본 모델 칸(3개)을 막지 않게 따로 (2026-11-09)
gemma_in_flight = threading.BoundedSemaphore(8)


def build_prompt(m: Loaded, history: list, text: str):
    """학습 때와 같은 형식으로 앞 대화를 붙인다. 길면 오래된 대화부터 뺀다 (PC 시험 페이지와 같은 방식)."""
    sp = m.sp
    budget = m.block_size - MAX_NEW_TOKENS
    recent = history[-PROMPT_HISTORY:] if PROMPT_HISTORY else []
    if recent and recent[0].role != "user":  # 사용자 말부터 시작하게
        recent = recent[1:]
    turns = [(t.role, " ".join(t.content.split())) for t in recent if t.content.strip()] + [("user", text)]
    while True:
        body = "\n".join(f"<|{role}|>\n{content}" for role, content in turns)
        ids = sp.encode(f"<|document_start|>\n{body}\n<|assistant|>\n", out_type=int)
        if len(ids) <= budget or len(turns) == 1:
            break
        turns = turns[2:] if len(turns) > 2 else turns[1:]
    if len(ids) > budget:  # 질문 하나만으로도 길면 질문 뒷부분을 자른다
        head = sp.encode("<|document_start|>\n<|user|>\n", out_type=int)
        tail = sp.encode("\n<|assistant|>\n", out_type=int)
        ids = head + sp.encode(text, out_type=int)[: max(1, budget - len(head) - len(tail))] + tail
    return ids, (len(turns) - 1) // 2


# 토크나이저가 영어를 소문자로 바꿔서 모델은 'ai'로만 쓴다 → 내보낼 때 'AI'로 (2026-10-19 사용자 요청)
AI_WORD = re.compile(r"(?<![A-Za-z])ai(?![A-Za-z])")


def fix_ai(text: str) -> str:
    return AI_WORD.sub("AI", text)


# =========================================================
# 가르친 대화 (PC 답변 서버의 /teach/pairs를 몇 분마다 받아 옴. PC가 꺼져 있으면 마지막으로 받은 것을 계속 씀)
# =========================================================
_TEACH_PUNCT = re.compile(r"[\s.,!?~…ㅠㅜ\"'()\[\]{}:;·\-]+")
taught: dict = {}
taught_info = {"count": 0, "updated": "", "at": 0.0}


def teach_norm(text: str) -> str:
    """같은 말인지 비교용: 띄어쓰기·문장 부호를 빼고 소문자로 (PC teach_store.norm과 같게)."""
    return _TEACH_PUNCT.sub("", text).lower()


taught_lock = threading.Lock()


def load_taught() -> bool:
    """PC 답변 서버에서 가르친 대화를 받아 온다. 성공하면 True (PC가 꺼져 있으면 False, 지난 것 그대로)."""
    with taught_lock:
        try:
            req = urllib.request.Request(TEACH_PAIRS_URL, headers={
                "User-Agent": "QueryBotAI/1.0 (https://querybot.kr/ai)", "ngrok-skip-browser-warning": "1"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
            new = {}
            for p in data.get("pairs", []):
                q, a = str(p.get("q", "")), str(p.get("a", ""))
                if q and a and len(q) <= MAX_INPUT_CHARS and len(a) <= MAX_INPUT_CHARS:
                    new[teach_norm(q)] = a  # 나중에 가르친 것이 앞의 것을 덮는다
            taught.clear()
            taught.update(new)
            taught_info.update(count=len(new), updated=time.strftime("%Y-%m-%d %H:%M:%S"), at=time.time())
            return True
        except Exception as exc:  # noqa: BLE001 — PC가 꺼져 있으면 실패
            log(f"가르친 대화 받기 실패(PC 꺼짐?): {type(exc).__name__}")
            return False


def refresh_taught() -> None:
    while True:
        load_taught()
        time.sleep(TEACH_REFRESH_SEC)


if TEACH_PAIRS_URL:
    threading.Thread(target=refresh_taught, daemon=True).start()


def make_reply(m: Loaded, history: list, text: str):
    ids, kept = build_prompt(m, history, text)
    banned = banned_ids(m.sp, text)  # 사용자 말에 숫자가 없으면 숫자로 새는 답과 깨진 글자를 막는다
    new = []
    with generate_lock:
        logits, past = m.model(ids)
        for step in range(MAX_NEW_TOKENS):
            token = pick_next(logits, step, new, m.space_id, TEMPERATURE, TOP_K, REPETITION, NO_REPEAT_NGRAM, banned)
            if token in m.stops:
                break
            new.append(token)
            logits, past = m.model([token], past)
    return fix_ai(m.sp.decode(new).strip()) or "...", kept


# =========================================================
# 웹 서버
# =========================================================
app = FastAPI(title="querybot AI", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=7200,
)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    # 본문을 읽기 전에 크기로 거른다. 길이 표시가 없는 POST(chunked)도 막는다.
    if request.method == "POST":
        length = request.headers.get("content-length", "")
        # /chat은 그림·파일 첨부(그림 4장 + 파일 글자 3만) 때문에 크게, 나머지는 그대로 작게 (2026-11-09)
        limit = MAX_CHAT_BODY_BYTES if request.url.path == "/chat" else MAX_BODY_BYTES
        if not length.isdigit() or int(length) > limit:
            return JSONResponse({"detail": "요청이 너무 큼"}, status_code=413)
    return await call_next(request)


recent_requests: dict = defaultdict(deque)
rate_lock = threading.Lock()


def client_ip(request: Request) -> str:
    # 방문자가 X-Forwarded-For를 지어 보내도, 프록시가 덧붙인 오른쪽 값은 못 바꾼다.
    parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if len(parts) >= PROXY_HOPS:
        return parts[-PROXY_HOPS]
    return request.client.host if request.client else "?"


def check_rate_limit(ip: str) -> None:
    now = time.time()
    with rate_lock:
        if len(recent_requests) > 10000:  # 오래 켜 둬도 메모리가 계속 늘지 않게
            for key in [k for k, q in recent_requests.items() if not q or now - q[-1] > RATE_WINDOW_SEC]:
                del recent_requests[key]
        q = recent_requests[ip]
        while q and now - q[0] > RATE_WINDOW_SEC:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="잠시 후 다시 시도")
        q.append(now)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=MAX_INPUT_CHARS * 2)


class FileIn(BaseModel):
    # 첨부 파일은 페이지가 글자로 읽어서 보낸다(PDF도 페이지에서 글자만 뽑음) (2026-11-09)
    name: str = Field(default="", max_length=120)
    text: str = Field(default="", max_length=40000)


class LocIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ChatIn(BaseModel):
    message: str = Field(max_length=MAX_INPUT_CHARS * 4)
    model: str = Field(default="latest", max_length=32)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY)
    # 그림 첨부: 페이지가 긴 변 1280px JPEG로 줄인 data URL (Gemma 4만, 2026-11-09 사용자 요청)
    images: list[str] = Field(default_factory=list, max_length=4)
    files: list[FileIn] = Field(default_factory=list, max_length=5)
    # 쿼리봇 자아 (2026-11-09): '깊게 생각' 단추, 이 브라우저에 저장된 사용자 기억(한 줄씩, Gemma 4만 씀)
    think: bool = False
    memory: list[str] = Field(default_factory=list, max_length=12)
    # 위치 공유 (2026-11-09): 페이지의 '위치' 단추를 켰을 때만, 대략 좌표(소수 둘째 자리). 저장·로그 안 함
    location: LocIn | None = None


IMAGE_URL = re.compile(r"^data:image/(jpeg|png|webp|gif);base64,[A-Za-z0-9+/=]+$")
MAX_IMAGE_CHARS = 2_500_000  # data URL 하나 2.5MB까지 (페이지는 보통 100~400KB로 줄여 보냄)
ATTACH_DEFAULT = "첨부한 내용을 살펴보고 설명해 줘"


def model_list() -> list:
    out = []
    for key, label, _ in CHOICES:
        item = {"key": key, "label": label}
        if key in loaded:
            item["info"] = loaded[key].info
        out.append(item)
    return out


@app.get("/")
async def root():
    return {"ok": True, "service": "querybot AI", "chat": "POST /chat"}


@app.get("/health")
@app.get("/state")  # 광고 차단 목록(Brave 실드 등)이 다른 사이트의 /health 요청을 막아서, 페이지는 이 주소를 쓴다
async def health():
    # async라 답 만드는 작업이 밀려 있어도 바로 응답한다 (켜져 있는데 '꺼짐'으로 보이지 않게).
    out = {"ok": True, "model": loaded["latest"].info, "models": model_list(), "memory_mb": memory_mb(),
           "taught": {"count": taught_info["count"], "updated": taught_info["updated"]}}
    if cf_chat.ready():
        out["quota"] = cf_chat.usage()  # Gemma 4 오늘 남은 뉴런 (무료 하루 10,000, 한국 오전 9시 초기화)
    return out


@app.get("/taught")
async def taught_list():
    # 이 사이트 서버가 지금 바로 답하는 데 쓰는 가르친 대화 (주인 페이지가 '사이트 반영됨'을 표시하는 데 씀)
    return {"updated": taught_info["updated"], "count": taught_info["count"],
            "items": [{"q": k, "a": v} for k, v in taught.items()]}


@app.post("/taught/refresh")
def taught_refresh(request: Request):
    # 주인이 가르친 직후 페이지가 부른다 → 5분을 기다리지 않고 바로 받아 옴 (너무 자주는 안 받음)
    check_rate_limit(client_ip(request))
    if time.time() - taught_info.get("at", 0) > 10:
        load_taught()
    return {"updated": taught_info["updated"], "count": taught_info["count"]}


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    if len(body.message) > MAX_INPUT_CHARS * 2:
        raise HTTPException(status_code=400, detail=f"{MAX_INPUT_CHARS}자 이하로")
    text = " ".join(body.message.split())
    attached = bool(body.images or body.files)
    if attached:
        if body.model != GEMMA_KEY:
            raise HTTPException(status_code=400, detail="그림·파일 첨부는 Gemma 4에서만 할 수 있어요")
        for u in body.images:
            if len(u) > MAX_IMAGE_CHARS or not IMAGE_URL.match(u):
                raise HTTPException(status_code=400, detail="그림 파일을 읽을 수 없어요 (jpg·png·webp·gif, 큰 그림은 줄여서)")
        if not text:
            text = ATTACH_DEFAULT  # 그림만 올리고 말없이 보내도 설명해 주기
    if not text:
        raise HTTPException(status_code=400, detail="빈 메시지")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail=f"{MAX_INPUT_CHARS}자 이하로")
    check_rate_limit(client_ip(request))
    slot = gemma_in_flight if body.model == GEMMA_KEY else in_flight
    if not slot.acquire(blocking=False):
        # 503이 아니라 429: 페이지는 503을 '서버 꺼짐'으로 보여 준다.
        raise HTTPException(status_code=429, detail="잠시 후 다시 시도")
    try:
        t0 = time.time()
        # 기본 모델로 대화할 때만: 주인이 가르친 말이면 그 답을 바로 (다른 모델을 고른 방문자에겐 그 모델 답 그대로)
        answer = taught.get(teach_norm(text)) if body.model == "latest" else None
        # 지식 질문 → 위키백과 요약(+출처). Gemma 4는 위키 첫 문장보다 스스로 더 잘 답해서 건너뜀(2026-11-08 사용자: "위키 때문에 멍청해 보인다")
        wiki = None if answer or body.model == GEMMA_KEY else wiki_search.lookup(text)
        extra = None  # Gemma 4 웹 검색 출처 {"sources": [...], "searched": 검색어}
        if answer:
            reply, kept = answer, 0
        elif wiki:
            reply, kept = wiki["reply"], 0
        elif body.model == GEMMA_KEY:
            if not cf_chat.ready():
                raise HTTPException(status_code=400, detail="없는 모델")
            try:
                # 그림·파일 첨부와 웹 검색은 Gemma 4만 (2026-11-09 사용자 요청)
                reply, extra = cf_chat.reply(text, body.history, body.images, body.files, think=body.think, memory=body.memory,
                                             location=body.location.model_dump() if body.location else None)
            except cf_chat.Busy:
                # 503이면 페이지가 서버 전체를 '꺼짐'으로 보여 준다 → 429(잠시 후 다시)로. 무료 사용량은 매일 오전 9시에 다시 채워짐
                q = cf_chat.usage()
                msg = ("오늘 Gemma 4 무료 사용량을 다 썼어요. 한국 시각 오전 9시에 다시 채워져요."
                       if q["left"] <= 0 else "Gemma 4가 잠시 바빠요. 조금 뒤에 다시 시도해 주세요.")
                raise HTTPException(status_code=429, detail=msg)
            kept = min(len(body.history), cf_chat.HISTORY_TURNS) // 2
        else:
            m = get_model(body.model)
            reply, kept = make_reply(m, body.history, text)
    finally:
        slot.release()
    kind = " (가르친 답)" if answer else " (위키백과)" if wiki else ""
    if attached:
        kind += f" [그림 {len(body.images)}·파일 {len(body.files)}]"
    if extra and extra.get("searched"):
        kind += f" [검색 {len(extra.get('sources', []))}건]"  # 검색어는 방문자가 쓴 말이라 로그에 남기지 않는다
    if extra and extra.get("thought"):
        kind += f" [생각 {extra['thought']['seconds']}초]"
    if extra and extra.get("remember"):
        kind += f" [기억 {len(extra['remember'])}줄]"  # 기억 내용은 남기지 않는다
    if body.location:
        kind += " [위치]"  # 좌표·동네 이름은 남기지 않는다
    log(f"[{body.model}] 질문 {len(text)}자, 앞 대화 {kept}턴 → 답 {len(reply)}자{kind} ({time.time() - t0:.1f}초)")
    out = {"reply": reply, "kept_turns": kept}
    if answer:
        out["taught"] = True
    if wiki:
        out["source"] = wiki["url"]
    if extra:
        # 페이지는 source가 있으면 '위키백과 답' 카드로 그리므로 웹 검색 출처는 sources(목록)로 따로 준다
        out["sources"] = extra.get("sources", [])[:10]
        if extra.get("searched"):
            out["searched"] = extra["searched"]
        if extra.get("thought"):
            out["thought"] = extra["thought"]  # {seconds, summary(한국어 요약), asked(사용자가 정한 초)}
        if extra.get("remember"):
            out["remember"] = extra["remember"]  # 페이지가 이 브라우저에 저장할 새 기억
    if body.model == GEMMA_KEY:
        out["quota"] = cf_chat.usage()  # 페이지가 '오늘 남은 사용량'을 바로 고침
    return out


if __name__ == "__main__":
    log(f"서버 시작: 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
