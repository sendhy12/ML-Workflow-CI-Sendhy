# Use Python 3.12 slim image for efficiency
FROM python:3.12.7-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p mlruns models logs

# Set permissions
RUN chmod +x modelling_advanced.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mlflow; print('MLflow OK')" || exit 1

# Default command
CMD ["python", "modelling_advanced.py"]