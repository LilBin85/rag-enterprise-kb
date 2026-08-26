FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Docling 的 PDF 表格模型会通过 OpenCV 导入本机动态库；slim 基础镜像不包含它们。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxcb1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY scripts/init_demo_principals.py ./scripts/init_demo_principals.py
COPY scripts/init_agent_checkpointer.py ./scripts/init_agent_checkpointer.py
# requirements.txt 是当前验证环境生成的完整锁；其中指定 CPU PyTorch 仓库。
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY migrations ./migrations

RUN addgroup --system ragkb && adduser --system --ingroup ragkb ragkb \
    && mkdir -p /app/data/local/objects /home/ragkb/.cache/huggingface \
    && chown -R ragkb:ragkb /app /home/ragkb

USER ragkb

CMD ["uvicorn", "rag_kb.main:app", "--host", "0.0.0.0", "--port", "8000"]
