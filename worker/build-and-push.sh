#!/bin/bash

# Build and push script for SAM3 RunPod worker
# Usage: ./build-and-push.sh [your-docker-username] [image-name] [tag]

set -e

DOCKER_USERNAME=${1:-"your-username"}
IMAGE_NAME=${2:-"sam3-mask-worker"}
TAG=${3:-"latest"}
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"
echo ""

# Build from the parent directory (so COPY worker/ works)
cd "$(dirname "$0")/.."

docker build -f worker/Dockerfile -t ${FULL_IMAGE_NAME} .

echo ""
echo "Build complete!"
echo ""
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing ${FULL_IMAGE_NAME} to Docker Hub..."
    docker push ${FULL_IMAGE_NAME}
    echo ""
    echo "✅ Image pushed successfully!"
    echo ""
    echo "Use this image in RunPod: ${FULL_IMAGE_NAME}"
else
    echo "Skipping push. Image is available locally as: ${FULL_IMAGE_NAME}"
fi

