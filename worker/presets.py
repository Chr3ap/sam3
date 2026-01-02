from dataclasses import dataclass

@dataclass(frozen=True)
class PostprocessPreset:
    min_region_area_px: int
    keep_largest_component: bool
    erode_px: int
    feather_px: int

PRESETS = {
    "fast": PostprocessPreset(
        min_region_area_px=1500,
        keep_largest_component=True,
        erode_px=2,
        feather_px=6,
    ),
    "balanced": PostprocessPreset(
        min_region_area_px=2000,
        keep_largest_component=True,
        erode_px=4,
        feather_px=12,
    ),
    "best": PostprocessPreset(
        min_region_area_px=3000,
        keep_largest_component=True,
        erode_px=6,
        feather_px=18,
    ),
}

