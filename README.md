# QueryBot AI (클라우드 답변 서버)

querybot.kr/ai 페이지의 답변 서버. 직접 학습한 작은 한국어 GPT 모델을 Render에서 돌린다.

- `GET /health` — 상태와 모델 정보
- `POST /chat` — `{"message": "안녕"}` → `{"reply": "..."}`

## 모델 바꾸기

PC에서 `python make_cloud_model.py <best_checkpoint.pth> <spm_small_chat_gpt.model>` →
`model_files/` 안의 파일(model.pt.part01, 02 ..., spm_small_chat_gpt.model)을 GitHub 맨 위에 다시 올리면
(조각 수가 줄었으면 남은 옛 조각은 지운다)
Render가 자동으로 새로 배포한다.
