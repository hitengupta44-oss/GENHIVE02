# ── HiveRift Intelligence Brain (Groq Edition) ────────────────────────────────
# Minimal image — no model downloads, no FAISS, no embeddings.
# LLM inference happens via Groq API at runtime.
# Required secret in HF Space: GROQ_API_KEY

FROM python:3.11-slim

# HuggingFace Spaces expects the app on port 7860
ENV PORT=7860

WORKDIR /app

# Install dependencies first (cached layer unless requirements.txt changes)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ ./backend/
COPY ["Knowledge Base/", "./Knowledge Base/"]
COPY frontend/ ./frontend/

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
