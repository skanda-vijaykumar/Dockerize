FROM python:3.12

WORKDIR /app

# Install system dependencies including git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip uninstall -y FlagEmbedding || true
RUN pip install git+https://github.com/FlagOpen/FlagEmbedding.git#egg=FlagEmbedding
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy project files
COPY *.py ./
COPY static/ ./static/
COPY templates/ ./templates/
COPY extracted_best/ ./extracted_best/
COPY reranker_patch.py ./

# Create directories if they don't exist
RUN mkdir -p static/js
RUN mkdir -p templates

# Add health endpoint for Docker
RUN echo 'from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get("/health")\ndef health():\n    return {"status": "ok"}' > health.py

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["bash", "-c", "uvicorn route18:app --host 0.0.0.0 --port 8000"]

