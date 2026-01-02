import base64
import io
import requests
import numpy as np
from PIL import Image

def load_image_rgb(image_spec: dict) -> np.ndarray:
    t = image_spec.get("type")
    v = image_spec.get("value")
    if t not in ("url", "b64"):
        raise ValueError("image.type must be 'url' or 'b64'")
    if not v:
        raise ValueError("image.value is required")

    if t == "url":
        r = requests.get(v, timeout=30)
        r.raise_for_status()
        data = r.content
    else:
        data = base64.b64decode(v)

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)

def png_alpha_from_mask(mask01: np.ndarray) -> bytes:
    """
    mask01: HxW float/bool in [0,1]. Produces RGBA PNG with alpha=mask.
    """
    mask = (mask01 * 255.0).clip(0, 255).astype(np.uint8)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 3] = mask
    img = Image.fromarray(rgba, mode="RGBA")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

