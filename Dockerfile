FROM python:3.12-slim
LABEL authors="Mohit"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 24500

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:24500/', timeout=3)" || exit 1

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:24500", "app:app"]