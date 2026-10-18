FROM python:3.12-slim

# Hugging Face Spaces는 1000번 사용자로 돌린다 (Render에서도 문제없음).
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py"]
