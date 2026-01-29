FROM python:3.9-slim

# Install system dependencies for building llama.cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# strict dependency versions prevent breakage
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    pydantic==2.5.3 \
    llama-cpp-python==0.2.26

# Set working directory
WORKDIR /app

# Copy model and code
COPY . .

# Expose port
EXPOSE 7860

# Run the API with high concurrency settings
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]