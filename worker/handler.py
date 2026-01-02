import time
import runpod
import numpy as np

from schema import validate_input
from errors import UserError, error_response
from presets import PRESETS
from image_io import load_image_rgb, png_alpha_from_mask, b64
from mask_postprocess import cleanup_mask, erode, feather
from overlay import overlay_mask
from sam3_adapter import SAM3Adapter
from checkpoint_loader import ensure_checkpoint

# --- Download checkpoint at runtime if needed (before model init) ---
print("Ensuring SAM3 checkpoint is available...")
ensure_checkpoint()

# --- Heavy init outside handler (RunPod best practice) ---
print("Initializing SAM3 model...")
SAM3 = SAM3Adapter(device="cuda")
print("SAM3 model initialized and ready!")

def _get_selection(job_input: dict):
    sel = job_input.get("selection") or {}
    mode = (sel.get("mode") or "text").lower()
    text = sel.get("text")
    # Future: box/points
    return mode, text

def _get_quality(job_input: dict):
    pp = job_input.get("postprocess") or {}
    quality = (pp.get("quality") or "balanced").lower()
    custom = pp.get("custom")
    if quality not in ("fast", "balanced", "best", "custom"):
        raise UserError("INVALID_INPUT", "postprocess.quality must be one of fast|balanced|best|custom")
    if quality == "custom" and not isinstance(custom, dict):
        raise UserError("INVALID_INPUT", "postprocess.custom must be provided when quality=custom")
    if quality != "custom" and custom is not None:
        # ignore, but don't fail
        custom = None
    return quality, custom

def _resolve_postprocess(quality: str, custom: dict | None):
    if quality != "custom":
        return PRESETS[quality]
    # Custom overrides with sane defaults (start from balanced)
    base = PRESETS["balanced"]
    return base.__class__(
        min_region_area_px=int(custom.get("min_region_area_px", base.min_region_area_px)),
        keep_largest_component=bool(custom.get("keep_largest_component", base.keep_largest_component)),
        erode_px=int(custom.get("erode_px", base.erode_px)),
        feather_px=int(custom.get("feather_px", base.feather_px)),
    )

def handler(job):
    t0 = time.time()

    try:
        job_input = job.get("input", {})
        validated = validate_input(job_input)
        if "errors" in validated:
            return error_response("INVALID_INPUT", "Validation failed", {"errors": validated["errors"]})
        inp = validated["validated_input"]  # per RunPod validator docs

        request_id = inp.get("request_id")
        image_spec = inp.get("image")
        target = (inp.get("target") or "").strip()

        mode, text = _get_selection(inp)
        if mode != "text":
            raise UserError("INVALID_INPUT", "Only selection.mode='text' is supported in v1")
        prompt_text = (text or target).strip()
        if not prompt_text:
            raise UserError("INVALID_INPUT", "selection.text or target must be non-empty")

        quality, custom = _get_quality(inp)
        pp = _resolve_postprocess(quality, custom)

        out_cfg = inp.get("output") or {}
        return_debug = bool(out_cfg.get("return_debug", True))
        return_instances = bool(out_cfg.get("return_instances", False))
        fmt = (out_cfg.get("format") or "png_alpha").lower()
        if fmt != "png_alpha":
            raise UserError("INVALID_INPUT", "Only output.format='png_alpha' is supported in v1")
        if return_instances:
            # Keep schema compatible, but v1 focuses on semantic mask
            return_instances = False

        # --- Load image
        t_img0 = time.time()
        rgb = load_image_rgb(image_spec)
        h, w = rgb.shape[:2]
        t_img1 = time.time()

        # --- SAM3 inference (mask candidates)
        t_sam0 = time.time()
        masks, scores = SAM3.segment_text(rgb, prompt_text)  # TODO in adapter
        t_sam1 = time.time()

        if not masks:
            raise UserError("NO_MASK", f"No masks returned for prompt '{prompt_text}'", {"prompt": prompt_text})

        # Check if we should combine multiple masks
        combine_masks = inp.get("combine_masks", False)

        if combine_masks and len(masks) > 1:
            # Combine all masks with union (OR operation) - keeps all floor regions
            raw = np.zeros_like(masks[0])
            for mask in masks:
                raw = np.maximum(raw, mask)  # Union: keep any pixel that's in any mask
            print(f"Combined {len(masks)} masks into single mask")
            best_i = 0  # For metadata, use first mask index
        else:
            # Choose best mask by score (original behavior)
            best_i = int(np.argmax(np.array(scores)))
            raw = masks[best_i]
            print(f"Using best mask (score: {scores[best_i]:.3f})")

        # --- Postprocess: hard + soft
        t_pp0 = time.time()
        hard01 = cleanup_mask(
            raw,
            min_region_area_px=pp.min_region_area_px,
            keep_largest_component=pp.keep_largest_component,
        )
        hard01 = erode(hard01, pp.erode_px).astype(np.float32)
        soft01 = feather(hard01, pp.feather_px).astype(np.float32)
        t_pp1 = time.time()

        # --- Encode outputs
        t_enc0 = time.time()
        hard_png = png_alpha_from_mask(hard01)
        soft_png = png_alpha_from_mask(soft01)
        debug_overlay = None
        if return_debug:
            ov = overlay_mask(rgb, hard01)
            # reuse png_alpha_from_mask by making a full RGBA? easier: encode RGB as PNG
            # minimal: return overlay as base64 PNG RGB via PIL
            from PIL import Image
            import io
            buf = io.BytesIO()
            Image.fromarray(ov, mode="RGB").save(buf, format="PNG")
            debug_overlay = buf.getvalue()
        t_enc1 = time.time()

        resp = {
            "ok": True,
            "request_id": request_id,
            "target": target,
            "image_size": {"width": int(w), "height": int(h)},
            "masks": {
                "semantic": {
                    "hard": {"encoding": "png_alpha", "png_b64": b64(hard_png)},
                    "soft": {"encoding": "png_alpha", "png_b64": b64(soft_png)},
                },
                "instances": [],
            },
            "debug": {"overlay_png_b64": b64(debug_overlay)} if debug_overlay else {},
            "meta": {
                "model": "sam3",
                "selection_mode": mode,
                "scores": [float(s) for s in scores],
                "num_masks": len(masks),
                "combine_masks": combine_masks,
                "postprocess_quality": quality,
                "timings_ms": {
                    "download_decode": int((t_img1 - t_img0) * 1000),
                    "sam_infer": int((t_sam1 - t_sam0) * 1000),
                    "postprocess": int((t_pp1 - t_pp0) * 1000),
                    "encode": int((t_enc1 - t_enc0) * 1000),
                    "total": int((time.time() - t0) * 1000),
                },
            },
        }
        return resp

    except UserError as e:
        return error_response(e.code, e.message, e.details)
    except Exception as e:
        # Let RunPod mark failures automatically if you prefer raising; here we return a stable error shape.
        return error_response("INTERNAL_ERROR", str(e), {})

runpod.serverless.start({"handler": handler})  # required

