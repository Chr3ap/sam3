# worker/sam3_adapter.py
from __future__ import annotations

import os
import numpy as np
import torch
from PIL import Image

class SAM3Adapter:
    """
    Adapter around facebookresearch/sam3.

    Design goals:
    - model loads once (constructed globally in handler.py)
    - provide one stable method: segment_text(image_rgb, text) -> (masks, scores)
    - hide SAM3 API churn behind this file
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.use_transformers = False  # Track which API we're using
        self._load()

    def _load(self):
        """
        Load SAM3 model and processor. Try Hugging Face transformers first, then fall back to direct SAM3 repo.
        """
        # Try Hugging Face transformers API first (if using HF model)
        try:
            from transformers import Sam3Model, Sam3Processor
            print("Loading SAM3 from Hugging Face transformers...")
            
            # Load model and processor from Hugging Face
            model_name = os.environ.get("SAM3_MODEL_NAME", "facebook/sam3")
            self.model = Sam3Model.from_pretrained(model_name).to(self.device)
            self.processor = Sam3Processor.from_pretrained(model_name)
            self.model.eval()
            self.use_transformers = True  # Mark that we're using transformers API
            print(f"Successfully loaded SAM3 from Hugging Face: {model_name}")
            return
        except ImportError:
            print("transformers not available, trying direct SAM3 repo...")
        except Exception as e:
            print(f"Hugging Face load failed: {e}, trying direct SAM3 repo...")
        
        # Fall back to direct SAM3 repo API
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            print("Loading SAM3 from direct repo...")
        except ImportError as e:
            raise RuntimeError(f"Failed importing SAM3 modules. Check sam3 install. Error: {e}")

        # ---- Configure checkpoint path via env (for direct repo method)
        ckpt = os.environ.get("SAM3_CHECKPOINT", "").strip()
        
        # If checkpoint path is provided, verify it exists
        # If not found, search for common checkpoint filenames
        if ckpt:
            if not os.path.exists(ckpt):
                # Try to find checkpoint in the checkpoints directory
                checkpoint_dir = os.path.dirname(ckpt) if os.path.dirname(ckpt) else "/app/checkpoints"
                if os.path.exists(checkpoint_dir):
                    # Look for common checkpoint filenames (check .pt and .pth files)
                    checkpoint_files = []
                    for root, dirs, files in os.walk(checkpoint_dir):
                        for file in files:
                            if file.endswith(('.pt', '.pth')) and 'sam3' in file.lower():
                                checkpoint_files.append(os.path.join(root, file))
                    
                    # Also check for exact common names
                    common_names = ["sam3.pt", "sam3_image_encoder.pth", "checkpoint.pth"]
                    for filename in common_names:
                        potential_path = os.path.join(checkpoint_dir, filename)
                        if os.path.exists(potential_path):
                            checkpoint_files.insert(0, potential_path)  # Prefer exact matches
                    
                    if checkpoint_files:
                        ckpt = checkpoint_files[0]  # Use first found
                        print(f"Found checkpoint at: {ckpt}")
                    else:
                        raise RuntimeError(f"SAM3 checkpoint file not found at: {ckpt} or in {checkpoint_dir}")
                else:
                    raise RuntimeError(f"SAM3 checkpoint file not found at: {ckpt}")
        else:
            # No checkpoint specified, try to find in default location
            checkpoint_dir = "/app/checkpoints"
            if os.path.exists(checkpoint_dir):
                for root, dirs, files in os.walk(checkpoint_dir):
                    for file in files:
                        if file.endswith(('.pt', '.pth')) and 'sam3' in file.lower():
                            ckpt = os.path.join(root, file)
                            print(f"Auto-detected checkpoint at: {ckpt}")
                            break
                    if ckpt:
                        break
        
        # Build model - build_sam3_image_model() may or may not accept checkpoint_path
        # Try different approaches based on the API
        try:
            if ckpt and os.path.exists(ckpt):
                # Try with checkpoint_path parameter first
                try:
                    self.model = build_sam3_image_model(checkpoint_path=ckpt)
                except TypeError:
                    # If checkpoint_path parameter doesn't exist, try setting it as default
                    # or the model might auto-detect from a default location
                    # Set checkpoint in environment or try loading without explicit path
                    self.model = build_sam3_image_model()
            else:
                # No checkpoint provided or not found, try auto-download/default location
                self.model = build_sam3_image_model()
        except Exception as e:
            raise RuntimeError(f"Failed to build SAM3 model. Checkpoint: {ckpt}. Error: {e}")
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize processor
        self.processor = Sam3Processor(self.model)
        print("Successfully loaded SAM3 from direct repo")

    @torch.inference_mode()
    def segment_text(self, image_rgb: np.ndarray, text: str):
        """
        Args:
          image_rgb: HxWx3 uint8 numpy array
          text: e.g. "floor", "cabinets", "walls"

        Returns:
          masks: list[np.ndarray] each HxW float32 in {0,1}
          scores: list[float]
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("SAM3 processor/model not initialized")

        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be HxWx3 numpy array")

        text = (text or "").strip()
        if not text:
            return [], []

        # Convert numpy array to PIL Image
        image = Image.fromarray(image_rgb, mode="RGB")

        # Use the appropriate API based on what was loaded
        if self.use_transformers:
            # Hugging Face transformers API
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=[image.size[::-1]]  # [height, width]
            )[0]
            
            # Extract masks and scores
            masks_raw = results.get("masks", [])
            scores_raw = results.get("scores", [])
        else:
            # Direct SAM3 repo API
            # Set image in processor
            inference_state = self.processor.set_image(image)

            # Apply text prompt
            output = self.processor.set_text_prompt(state=inference_state, prompt=text)

            # Extract masks and scores from output
            masks_raw = output.get("masks", [])
            scores_raw = output.get("scores", [])

        # Convert masks to numpy arrays and normalize to [0,1]
        masks = []
        scores = []
        
        for i, mask in enumerate(masks_raw):
            # Handle different mask formats (tensors, numpy arrays, etc.)
            if torch.is_tensor(mask):
                mask_np = mask.cpu().numpy()
            elif isinstance(mask, np.ndarray):
                mask_np = mask
            else:
                continue
            
            # Ensure mask is 2D HxW
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            # Normalize to [0,1] if needed
            if mask_np.max() > 1.0:
                mask_np = mask_np.astype(np.float32) / 255.0
            else:
                mask_np = mask_np.astype(np.float32)
            
            # Ensure values are in [0,1] range
            mask_np = np.clip(mask_np, 0.0, 1.0)
            
            masks.append(mask_np)
            
            # Extract score if available
            if i < len(scores_raw):
                score = scores_raw[i]
                if torch.is_tensor(score):
                    score = score.item()
                scores.append(float(score))
            else:
                # Default score if not provided
                scores.append(1.0)

        return masks, scores

def sanity_device():
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda"

