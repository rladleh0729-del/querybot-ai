"""querybot.kr/ai 페이지용 답변 서버 (클라우드판).

PC판(ai_server.py)과 같은 일을 하지만 ngrok 없이 클라우드(Render)에서 24시간 돈다.
포트는 환경변수 PORT를 따르고(Render), 없으면 7860.

모델 파일 (PC에서는 model_files/ 안, GitHub에서는 app.py 옆 — 웹 업로드는 폴더를 못 만들어서):
  model.pt.part01, 02 ...         ← 기본(최신) 모델. make_cloud_model.py로 줄여서 9MB씩 자른 체크포인트
  spm_small_chat_gpt.model        ← 기본 모델의 토크나이저
  <이름>.model.pt.part01 ...      ← 방문자가 골라 볼 수 있는 다른 모델 (EXTRA_MODELS)
  <이름>.spm.model                   (make_cloud_model.py --prefix <이름> --int8)

방문자는 모델을 고르고, '이전 대화 기억'을 켜면 앞 대화(history)를 같이 보낸다.
서버는 대화를 저장하지 않는다 — 페이지가 들고 있다가 매번 보낸다.
"""
import gc
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai_sampling import banned_ids, pick_next, space_id_of
from model import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("AI_MODEL_DIR", str(
    BASE_DIR / "model_files" if (BASE_DIR / "model_files").is_dir() else BASE_DIR)))

# 고를 수 있는 다른 모델: (키, 이름, 파일 앞머리). 기본 모델은 "latest".
# Render 무료(메모리 512MB)에 같이 들어가도록 처음 고를 때 불러온다. 파일은 PC에서 미리 8비트로
# 줄여 둔다(make_cloud_model.py --int8, 답은 거의 같음). 안 줄인 파일이면 불러온 뒤 줄인다.
EXTRA_MODELS = [
    ("stage3", "3단계 모델", "stage3"),
    ("v1", "첫 모델 (1,380만)", "v1"),
]

PORT = int(os.environ.get("PORT", "7860"))
torch.set_num_threads(max(1, int(os.environ.get("AI_CPU_THREADS", str(os.cpu_count() or 1)))))

MAX_INPUT_CHARS = 200
MAX_HISTORY = 20         # 앞 대화는 최근 20개(10번 주고받기)까지만 받는다
MAX_BODY_BYTES = 16384   # 요청 본문 상한 (질문 200자 + 앞 대화 20개)
MAX_NEW_TOKENS = 120     # 정상 답은 대개 짧지만, 2~3문장 답이 잘리지 않게 넉넉히
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
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================
# 모델 불러오기
# =========================================================
def model_files(prefix: str):
    """(체크포인트 경로, 토크나이저 경로). 기본 모델은 앞머리가 없다."""
    if prefix:
        return MODEL_DIR / f"{prefix}.model.pt", MODEL_DIR / f"{prefix}.spm.model"
    return MODEL_DIR / "model.pt", MODEL_DIR / "spm_small_chat_gpt.model"


def has_files(prefix: str) -> bool:
    ckpt, spm_path = model_files(prefix)
    return (ckpt.exists() or any(MODEL_DIR.glob(ckpt.name + ".part*"))) and spm_path.exists()


class Loaded:
    def __init__(self, prefix: str, quantize: bool):
        ckpt_path, spm_path = model_files(prefix)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(spm_path))
        model, ckpt = load_model(ckpt_path)
        pre_quantized = "params" in ckpt  # make_cloud_model.py --int8 로 미리 줄인 모델
        self.block_size = model.config.block_size
        self.info = {
            "params": int(ckpt["params"]) if pre_quantized else sum(p.numel() for p in model.parameters()),
            "iter": int(ckpt.get("iter_num", 0)),
            "val_loss": round(float(ckpt.get("best_val_loss", 0.0)), 3),
            "saved_at": ckpt.get("saved_at", ""),
        }
        del ckpt
        if quantize and not pre_quantized:
            model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        self.model = model
        unk = self.sp.unk_id()
        self.stops = {i for i in (self.sp.piece_to_id(s) for s in SPECIALS) if i != unk}
        self.space_id = space_id_of(self.sp)  # 첫 토큰 빈칸 막기용 (ai_sampling.py 설명 참고)


if not has_files(""):
    raise SystemExit(f"기본 모델 파일이 없음: {MODEL_DIR}")
loaded = {"latest": Loaded("", quantize=False)}
gc.collect()
log(f"기본 모델 준비: {loaded['latest'].info}")
CHOICES = [("latest", "최신 모델", "")] + [m for m in EXTRA_MODELS if has_files(m[2])]
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
            loaded[key] = Loaded(prefix, quantize=True)
            gc.collect()
            log(f"모델 불러옴: {key} ({time.time() - t0:.1f}초) {loaded[key].info} 메모리 {memory_mb()}")
        return loaded[key]


generate_lock = threading.Lock()
in_flight = threading.BoundedSemaphore(MAX_IN_FLIGHT)


def build_prompt(m: Loaded, history: list, text: str):
    """학습 때와 같은 형식으로 앞 대화를 붙인다. 길면 오래된 대화부터 뺀다 (PC 시험 페이지와 같은 방식)."""
    sp = m.sp
    budget = m.block_size - MAX_NEW_TOKENS
    turns = [(t.role, " ".join(t.content.split())) for t in history if t.content.strip()] + [("user", text)]
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


def make_reply(m: Loaded, history: list, text: str):
    ids, kept = build_prompt(m, history, text)
    banned = banned_ids(m.sp, text)  # 사용자 말에 숫자가 없으면 숫자로 새는 답과 깨진 글자를 막는다
    new = []
    with generate_lock, torch.inference_mode():
        logits, past = m.model(torch.tensor([ids], dtype=torch.long))
        for step in range(MAX_NEW_TOKENS):
            nxt = pick_next(logits, step, new, m.space_id, TEMPERATURE, TOP_K, REPETITION, NO_REPEAT_NGRAM, banned)
            token = int(nxt.item())
            if token in m.stops:
                break
            new.append(token)
            logits, past = m.model(nxt.view(1, 1), past)
    return m.sp.decode(new).strip() or "...", kept


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
        if not length.isdigit() or int(length) > MAX_BODY_BYTES:
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


class ChatIn(BaseModel):
    message: str = Field(max_length=MAX_INPUT_CHARS * 4)
    model: str = Field(default="latest", max_length=32)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY)


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
    return {"ok": True, "model": loaded["latest"].info, "models": model_list(), "memory_mb": memory_mb()}


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    if len(body.message) > MAX_INPUT_CHARS * 2:
        raise HTTPException(status_code=400, detail=f"{MAX_INPUT_CHARS}자 이하로")
    text = " ".join(body.message.split())
    if not text:
        raise HTTPException(status_code=400, detail="빈 메시지")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail=f"{MAX_INPUT_CHARS}자 이하로")
    check_rate_limit(client_ip(request))
    if not in_flight.acquire(blocking=False):
        # 503이 아니라 429: 페이지는 503을 '서버 꺼짐'으로 보여 준다.
        raise HTTPException(status_code=429, detail="잠시 후 다시 시도")
    try:
        t0 = time.time()
        m = get_model(body.model)
        reply, kept = make_reply(m, body.history, text)
    finally:
        in_flight.release()
    log(f"[{body.model}] 질문 {len(text)}자, 앞 대화 {kept}턴 → 답 {len(reply)}자 ({time.time() - t0:.1f}초)")
    return {"reply": reply, "kept_turns": kept}


if __name__ == "__main__":
    log(f"서버 시작: 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
