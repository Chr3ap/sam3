"""
Utility to download SAM3 checkpoint at runtime if not present.
This allows the Docker image to stay small and download the checkpoint on first use.
"""
import os
from huggingface_hub import hf_hub_download

def ensure_checkpoint():
    """
    Ensure SAM3 checkpoint exists. Download from Hugging Face if missing.
    Uses HF_TOKEN environment variable for authentication.
    """
    checkpoint_path = os.environ.get("SAM3_CHECKPOINT", "/app/checkpoints/sam3.pt")
    checkpoint_dir = os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else "/app/checkpoints"
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if checkpoint already exists
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint already exists at: {checkpoint_path}")
        return checkpoint_path
    
    # Check for other common checkpoint filenames
    common_names = ["sam3.pt", "sam3_image_encoder.pth", "checkpoint.pth"]
    for filename in common_names:
        potential_path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(potential_path):
            print(f"Found checkpoint at: {potential_path}")
            # Update environment variable to point to found checkpoint
            os.environ["SAM3_CHECKPOINT"] = potential_path
            return potential_path
    
    # Checkpoint not found, download it
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable required to download checkpoint. "
            "Please set HF_TOKEN in your RunPod endpoint environment variables."
        )
    
    print("Checkpoint not found. Downloading SAM3 checkpoint from Hugging Face...")
    print("This may take a few minutes on first run...")
    
    try:
        # Download checkpoint file
        # hf_hub_download returns the full path to the downloaded file
        downloaded_path = hf_hub_download(
            repo_id="facebook/sam3",
            filename="sam3.pt",
            local_dir=checkpoint_dir,
            token=hf_token
        )
        
        # Also download config.json if needed (optional)
        try:
            hf_hub_download(
                repo_id="facebook/sam3",
                filename="config.json",
                local_dir=checkpoint_dir,
                token=hf_token
            )
        except Exception as e:
            print(f"Warning: Could not download config.json: {e}")
        
        # Verify the downloaded file exists
        if os.path.exists(downloaded_path):
            os.environ["SAM3_CHECKPOINT"] = downloaded_path
            print(f"Checkpoint downloaded successfully to: {downloaded_path}")
            return downloaded_path
        else:
            # Fallback: check if it's in the expected location
            expected_path = os.path.join(checkpoint_dir, "sam3.pt")
            if os.path.exists(expected_path):
                os.environ["SAM3_CHECKPOINT"] = expected_path
                print(f"Checkpoint found at expected location: {expected_path}")
                return expected_path
            else:
                raise RuntimeError(f"Downloaded checkpoint not found at {downloaded_path} or {expected_path}")
            
    except Exception as e:
        raise RuntimeError(
            f"Failed to download SAM3 checkpoint from Hugging Face. "
            f"Make sure HF_TOKEN is set correctly. Error: {e}"
        )

