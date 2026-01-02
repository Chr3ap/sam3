import numpy as np
import cv2

def _largest_component(binary: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if num <= 1:
        return binary
    # component 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8)

def cleanup_mask(mask: np.ndarray, min_region_area_px: int, keep_largest_component: bool) -> np.ndarray:
    """
    mask: HxW float/bool
    returns HxW uint8 {0,1}
    """
    m = (mask > 0.5).astype(np.uint8)

    # remove small components by area threshold
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_region_area_px:
            keep[labels == i] = 1

    if keep_largest_component:
        keep = _largest_component(keep)

    return keep

def erode(binary01: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return binary01
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.erode(binary01.astype(np.uint8), k, iterations=1)

def feather(binary01: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return binary01.astype(np.float32)
    # Gaussian blur on a 0/1 mask yields a soft matte
    k = max(3, int(px) * 2 + 1)
    if k % 2 == 0:
        k += 1
    soft = cv2.GaussianBlur(binary01.astype(np.float32), (k, k), sigmaX=0, sigmaY=0)
    # normalize to [0,1]
    soft = np.clip(soft, 0.0, 1.0)
    return soft

