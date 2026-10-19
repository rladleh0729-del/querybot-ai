FROM python:3.12-slim

# Hugging Face Spaces는 1000번 사용자로 돌린다 (Render에서도 문제없음).
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV PYTHONUNBUFFERED=1
# 메모리 할당 구역을 줄여서 Render 무료(512MB) 안에서 버티게 한다
ENV MALLOC_ARENA_MAX=2
EXPOSE 7860
CMD ["python", "app.py"]
