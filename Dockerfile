FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LOCALSCRIPT_MODEL=qwen2.5-coder:7b \
    OLLAMA_HOST=http://ollama:11434 \
    LUAC_BIN=/usr/bin/luac5.4

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends lua5.4 ca-certificates \
    && ln -sf /usr/bin/lua5.4 /usr/local/bin/lua \
    && ln -sf /usr/bin/luac5.4 /usr/local/bin/luac \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
