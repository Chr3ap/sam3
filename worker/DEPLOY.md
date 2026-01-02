# Deploying SAM3 Worker to RunPod

## Prerequisites

- Docker installed and running
- Docker Hub account (or another container registry)
- RunPod account with API access

## Building and Pushing the Docker Image

### Option 1: Using the build script

1. Make the script executable:
```bash
chmod +x worker/build-and-push.sh
```

2. Run the script with your Docker Hub username:
```bash
./worker/build-and-push.sh your-docker-username sam3-mask-worker latest
```

Replace:
- `your-docker-username` with your Docker Hub username
- `sam3-mask-worker` with your desired image name (optional)
- `latest` with your desired tag (optional, defaults to latest)

### Option 2: Manual build and push

1. Navigate to the project root directory:
```bash
cd /path/to/pipe
```

2. Build the Docker image:
```bash
docker build -f worker/Dockerfile -t your-username/sam3-mask-worker:latest .
```

3. Login to Docker Hub:
```bash
docker login
```

4. Push the image:
```bash
docker push your-username/sam3-mask-worker:latest
```

## Setting Up RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)

2. Create a new Serverless endpoint

3. Configure the endpoint:
   - **Container Image**: `your-username/sam3-mask-worker:latest`
   - **Container Disk**: At least 20GB (for model checkpoints)
   - **GPU Type**: Select based on your needs (e.g., RTX 3090, A100)

4. Set Environment Variables:
   - `SAM3_CHECKPOINT`: Path to your SAM3 checkpoint file (e.g., `/workspace/sam3_checkpoint.pth`)
   - `SAM3_CONFIG`: SAM3 config name (if required by the model)

5. **Important**: You'll need to either:
   - Mount the checkpoint file via RunPod's volume system, OR
   - Download it during container startup (add to Dockerfile or use init script)

## Downloading SAM3 Checkpoints

SAM3 checkpoints need to be requested from Meta. You can:

1. Request access from the [SAM3 repository](https://github.com/facebookresearch/sam3)
2. Download checkpoints to a RunPod volume
3. Or modify the Dockerfile to download during build (if you have access)

Example Dockerfile addition (if you have checkpoint access):
```dockerfile
# Add checkpoint download (requires authentication)
RUN pip install huggingface_hub
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='facebook/sam3', filename='sam3_image_encoder.pth', local_dir='/app/checkpoints')"
```

## Testing Locally

Before deploying, test locally with RunPod's test mode:

```bash
runpod test --template worker/test_input.json
```

Or use the RunPod CLI:
```bash
runpod serverless start --template worker/test_input.json
```

## Environment Variables in RunPod

When creating your endpoint, make sure to set:
- `SAM3_CHECKPOINT`: Full path to checkpoint file
- `SAM3_CONFIG`: Config name (check SAM3 docs for exact format)

## Troubleshooting

- **Import errors**: Make sure SAM3 is installed correctly in the Dockerfile
- **Checkpoint not found**: Verify the `SAM3_CHECKPOINT` path and that the file exists in the container
- **CUDA errors**: Ensure the base image CUDA version matches your GPU drivers

