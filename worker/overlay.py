import numpy as np

def overlay_mask(rgb: np.ndarray, mask01: np.ndarray, color=(0, 255, 0), alpha=0.35) -> np.ndarray:
    """
    rgb: HxWx3 uint8
    mask01: HxW float/bool
    """
    img = rgb.copy().astype(np.float32)
    m = (mask01 > 0.5).astype(np.float32)[..., None]
    col = np.array(color, dtype=np.float32)[None, None, :]
    img = img * (1.0 - m * alpha) + col * (m * alpha)
    return img.clip(0, 255).astype(np.uint8)

