FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (changes infrequently - cache this layer)
COPY worker/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip cache purge

# Install SAM3 from your GitHub fork (changes infrequently - cache this layer)
# Install directly from git - this is more reliable than editable installs
# Update this to your fork: Chr3ap/sam3
RUN pip install git+https://github.com/Chr3ap/sam3.git && \
    pip cache purge

# Install Hugging Face CLI which automatically handles git-xet for large files
RUN pip install huggingface_hub && \
    curl -LsSf https://hf.co/cli/install.sh | bash && \
    pip cache purge

# Add HF CLI to PATH (use /root since we're running as root in Docker)
ENV PATH="/root/.local/bin:$PATH"

# Create checkpoints directory (checkpoint will be downloaded at runtime)
RUN mkdir -p /app/checkpoints

# Set default checkpoint path (can be overridden by environment variable)
# Checkpoint will be downloaded at runtime by checkpoint_loader.py
ENV SAM3_CHECKPOINT=/app/checkpoints/sam3.pt

# COPY CODE LAST (changes frequently - this invalidates minimal layers)
# Only this layer needs to be rebuilt/pushed when code changes
COPY worker /app

CMD ["python3", "handler.py"]

