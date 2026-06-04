# JointBERT SLU serving image.
#
# Bakes in code + dependencies only; the trained model (~400MB, gitignored) is
# mounted at runtime from ./artifacts so the image stays small and reproducible.
# Train once with `python -m src.slu.train` (or mount an existing artifact dir),
# then `docker compose up`.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; torch/transformers ship their own wheels.
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SLU_ARTIFACT_DIR=/app/artifacts/jointbert \
    HF_HOME=/app/.hf_cache

# Install dependencies first for layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-bake the base encoder + tokenizer so the container runs offline. The mounted
# model.pt holds only the fine-tuned state_dict; without this, the first request
# would pull ~440MB of bert-base-uncased from the HuggingFace Hub at runtime (and
# fail in an air-gapped deploy). Baking it here makes the image self-sufficient.
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('bert-base-uncased'); \
    AutoTokenizer.from_pretrained('bert-base-uncased')"

# App code.
COPY src/ ./src/

EXPOSE 8000

# Single worker: the model holds ~400MB; scale with replicas, not workers.
CMD ["uvicorn", "src.slu.api:app", "--host", "0.0.0.0", "--port", "8000"]
