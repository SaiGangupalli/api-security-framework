# Stage 1: Train models
FROM python:3.9-slim as model-trainer

WORKDIR /app

# Install dependencies first
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    wheel==0.40.0 \
    setuptools==65.5.1 \
    numpy==1.21.6 \
    scipy==1.7.3 \
    pandas==1.3.5 \
    scikit-learn==1.0.2 \
    joblib==1.1.1 \
    pyarrow==14.0.1

# Copy source code
COPY src /app/src

# Create models directory
RUN mkdir -p /app/src/ml/models

# Set Python path and train models
ENV PYTHONPATH=/app/src
RUN python -u /app/src/ml/train.py

# Verify models were created
RUN ls -la /app/src/ml/models/

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    wheel==0.40.0 \
    setuptools==65.5.1 \
    numpy==1.21.6 \
    scipy==1.7.3 \
    pandas==1.3.5 \
    scikit-learn==1.0.2 \
    joblib==1.1.1 \
    pyarrow==14.0.1

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src /app/src

# Create models directory
RUN mkdir -p /app/src/ml/models

# Copy trained models from builder stage
COPY --from=model-trainer /app/src/ml/models/security_models.joblib /app/src/ml/models/
COPY --from=model-trainer /app/src/ml/models/model_metadata.joblib /app/src/ml/models/
COPY --from=model-trainer /app/src/ml/models/model_performance.joblib /app/src/ml/models/

# Verify copied models
RUN ls -la /app/src/ml/models/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]