"""querybot.kr/ai 페이지용 답변 서버 (클라우드판).

PC판(ai_server.py)과 같은 일을 하지만 ngrok 없이 클라우드(Hugging Face Spaces, Render 등)에서
24시간 돈다. 포트는 환경변수 PORT를 따르고(Render), 없으면 7860(Hugging Face).

모델 파일:
  model.pt.part01, 02 ...   ← make_cloud_model.py로 줄여서 9MB씩 자른 체크포인트
  spm_small_chat_gpt.model  ← 같은 단계의 토크나이저
  (PC에서는 model_files/ 안, GitHub에서는 app.py 옆)
"""
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import sentencepiece as spm
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai_sampling import pick_next, space_id_of
from model import load_model

BASE_DIR = Path(__file__).resolve().parent
# PC에서는 model_files/ 안에, GitHub(웹 업로드라 폴더를 못 만듦)에서는 app.py 옆에 있다.
MODEL_DIR = Path(os.environ.get("AI_MODEL_DIR", str(
    BASE_DIR / "model_files" if (BASE_DIR / "model_files").is_dir() else BASE_DIR)))
CKPT_PATH = MODEL_DIR / "model.pt"
SPM_PATH = MODEL_DIR / "spm_small_chat_gpt.model"

PORT = int(os.environ.get("PORT", "7860"))
torch.set_num_threads(max(1, int(os.environ.get("AI_CPU_THREADS", str(os.cpu_count() or 1)))))

MAX_INPUT_CHARS = 200
MAX_BODY_BYTES = 4096    # 요청 본문 상한 (200자 한글 JSON은 1KB 안팎)
MAX_NEW_TOKENS = 120     # 정상 답은 대개 짧지만, 2~3문장 답이 잘리지 않게 넉넉히
MAX_IN_FLIGHT = 3        # 답 만드는 중 1개 + 기다리는 2개까지. 넘치면 바로 "잠시 후"
TEMPERATURE = 0.7
TOP_K = 40
REPETITION = 1.1         # 반복 벌점 (약하게. 세면 조사·어미가 망가짐)
NO_REPEAT_NGRAM = 3      # 이번 답에 나온 3조각 묶음은 다시 못 씀
RATE_LIMIT = 20          # 한 사람(IP)당
RATE_WINDOW_SEC = 60     # 이 시간 안에 최대 RATE_LIMIT번
# 클라우드 앞단 프록시가 X-Forwarded-For 뒤에 덧붙이는 값 개수. 오른쪽에서 이만큼 들어간 값이 진짜 접속 주소.
PROXY_HOPS = max(1, int(os.environ.get("AI_PROXY_HOPS", "1")))

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
if not (CKPT_PATH.exists() or any(MODEL_DIR.glob(CKPT_PATH.name + ".part*"))) or not SPM_PATH.exists():
    raise SystemExit(f"모델 파일이 없음:\n  {CKPT_PATH}(.partNN)\n  {SPM_PATH}")

sp = spm.SentencePieceProcessor()
sp.load(str(SPM_PATH))
model, ckpt = load_model(CKPT_PATH)
MODEL_INFO = {
    "params": sum(p.numel() for p in model.parameters()),
    "iter": int(ckpt.get("iter_num", 0)),
    "val_loss": round(float(ckpt.get("best_val_loss", 0.0)), 3),
    "saved_at": ckpt.get("saved_at", ""),
}
del ckpt
log(f"모델 준비: 파라미터 {MODEL_INFO['params']:,}개, 학습 {MODEL_INFO['iter']:,}번째, val loss {MODEL_INFO['val_loss']}")

generate_lock = threading.Lock()
in_flight = threading.BoundedSemaphore(MAX_IN_FLIGHT)

END_ID = sp.piece_to_id("<|document_end|>")
USER_ID = sp.piece_to_id("<|user|>")  # 여러 턴 대화로 배운 모델은 답 뒤에 이 표시를 내놓기도 한다
STOP_IDS = {END_ID, USER_ID}
PROMPT_HEAD = sp.encode("<|document_start|>\n<|user|>\n", out_type=int)
PROMPT_TAIL = sp.encode("\n<|assistant|>\n", out_type=int)
SPACE_ID = space_id_of(sp)  # 첫 토큰 빈칸 막기용 (ai_sampling.py 설명 참고)


def make_reply(text: str) -> str:
    # 질문 부분만 잘라서 머리말·꼬리말(학습 때 형식)은 항상 남긴다. 질문+답이 block_size를 넘지 않는다.
    budget = model.config.block_size - MAX_NEW_TOKENS - len(PROMPT_HEAD) - len(PROMPT_TAIL)
    ids = PROMPT_HEAD + sp.encode(text, out_type=int)[:budget] + PROMPT_TAIL
    new = []
    with generate_lock, torch.inference_mode():
        logits, past = model(torch.tensor([ids], dtype=torch.long))
        for step in range(MAX_NEW_TOKENS):
            nxt = pick_next(logits, step, new, SPACE_ID, TEMPERATURE, TOP_K, REPETITION, NO_REPEAT_NGRAM)
            token = int(nxt.item())
            if token in STOP_IDS:
                break
            new.append(token)
            logits, past = model(nxt.view(1, 1), past)
    return sp.decode(new).strip() or "..."


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


class ChatIn(BaseModel):
    message: str = Field(max_length=MAX_INPUT_CHARS * 4)


@app.get("/")
async def root():
    return {"ok": True, "service": "querybot AI", "chat": "POST /chat"}


@app.get("/health")
@app.get("/state")  # 광고 차단 목록(Brave 실드 등)이 다른 사이트의 /health 요청을 막아서, 페이지는 이 주소를 쓴다
async def health():
    # async라 답 만드는 작업이 밀려 있어도 바로 응답한다 (켜져 있는데 '꺼짐'으로 보이지 않게).
    return {"ok": True, "model": MODEL_INFO}


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
        reply = make_reply(text)
    finally:
        in_flight.release()
    log(f"질문 {len(text)}자 → 답 {len(reply)}자 ({time.time() - t0:.1f}초)")
    return {"reply": reply}


if __name__ == "__main__":
    log(f"서버 시작: 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
