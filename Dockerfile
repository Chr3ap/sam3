FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps
COPY worker/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip cache purge

# Install SAM3 from current repo
RUN pip install -e . && \
    pip cache purge

# Install Hugging Face hub
RUN pip install huggingface_hub && \
    pip cache purge

# Create checkpoints directory
RUN mkdir -p /app/checkpoints
ENV SAM3_CHECKPOINT=/app/checkpoints/sam3.pt

# Copy everything (including worker code)
COPY . /app

# Run the worker
CMD ["python3", "worker/handler.py"]
