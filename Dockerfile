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

COPY pyproject.toml README.md ./
COPY src ./src
# 生产只做 CPU 推理；先从 PyTorch 官方 CPU 仓库安装，避免 PyPI 默认轮子拉取 CUDA 运行时。
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.13,<3" "torchvision>=0,<1"
# worker 会真实解析 PDF 并生成 bge-m3 向量，不能只安装基础 API 依赖。
RUN python -m pip install --no-cache-dir ".[models]"

COPY alembic.ini ./
COPY migrations ./migrations

RUN addgroup --system ragkb && adduser --system --ingroup ragkb ragkb \
    && mkdir -p /app/data/local/objects /home/ragkb/.cache/huggingface \
    && chown -R ragkb:ragkb /app /home/ragkb

USER ragkb

CMD ["uvicorn", "rag_kb.main:app", "--host", "0.0.0.0", "--port", "8000"]
