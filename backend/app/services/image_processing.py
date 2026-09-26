"""Image preprocessing pipeline for microplastic micrographs.

Pipeline: resize (if needed) -> grayscale -> denoise -> contrast enhancement
(optional CLAHE) -> thresholding -> morphological opening/closing.

The result is a binary mask in which candidate particles are white (255) on a
black background. All thresholds/kernels come from ``app.config.settings`` so
experiments can be tuned without touching the code.

Scientific note: Otsu's method automatically picks a global threshold by
minimizing intra-class variance, which works well for bright-field micrographs
where particles are darker than the filter background. ``THRESH_BINARY_INV``
makes dark particles become white foreground pixels in the mask.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.config import Settings, settings

PreprocessResult = dict[str, Any]


def _ensure_odd(value: int, minimum: int = 3) -> int:
    """OpenCV blur/morphology kernel sizes must be odd (and >= minimum)."""
    if value < minimum:
        value = minimum
    return value if value % 2 == 1 else value + 1


def resize_if_necessary(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, bool]:
    """Downscale very large images (memory safety). Returns (image, resized?)."""
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return image, False
    scale = max_dimension / float(largest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, True


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale (pass through if already single channel)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, method: str, gaussian_k: int, median_k: int) -> np.ndarray:
    """Reduce sensor noise before thresholding."""
    if method == "median":
        return cv2.medianBlur(gray, _ensure_odd(median_k, 3))
    kernel = _ensure_odd(gaussian_k, 3)
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def enhance_contrast(
    gray: np.ndarray, enabled: bool, clip_limit: float, tile_grid_size: int
) -> np.ndarray:
    """Optionally apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if not enabled:
        return gray
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(_ensure_odd(tile_grid_size, 3), _ensure_odd(tile_grid_size, 3)),
    )
    return clahe.apply(gray)


def binarize(gray: np.ndarray, method: str, invert: bool, block_size: int) -> np.ndarray:
    """Threshold the denoised grayscale image into a particle mask."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "adaptive":
        # Adaptive threshold handles uneven illumination across the field of view.
        block = _ensure_odd(block_size, 3)
        _, binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_GAUSSIAN_C, flag, block, 5
        )
        return binary
    _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return binary


def morphological_cleanup(
    binary: np.ndarray, open_k: int, close_k: int, iterations: int
) -> np.ndarray:
    """Morphological opening removes speckle noise; closing seals particle holes."""
    open_kernel = np.ones(
        (_ensure_odd(open_k, 1), _ensure_odd(open_k, 1)), np.uint8
    )
    close_kernel = np.ones(
        (_ensure_odd(close_k, 1), _ensure_odd(close_k, 1)), np.uint8
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
    return cleaned


def preprocess(image_bgr: np.ndarray, config: Settings | None = None) -> PreprocessResult:
    """Run the full preprocessing pipeline and return all intermediate images.

    The uploaded image is never modified: OpenCV operations return new arrays.
    """
    cfg = config or settings
    resized, was_resized = resize_if_necessary(image_bgr, cfg.MAX_PROCESSING_DIMENSION)
    gray = to_grayscale(resized)
    blurred = denoise(gray, cfg.BLUR_METHOD, cfg.GAUSSIAN_KERNEL_SIZE, cfg.MEDIAN_KERNEL_SIZE)
    enhanced = enhance_contrast(
        blurred, cfg.CLAHE_ENABLED, cfg.CLAHE_CLIP_LIMIT, cfg.CLAHE_TILE_GRID_SIZE
    )
    binary = binarize(enhanced, cfg.THRESHOLD_METHOD, cfg.THRESHOLD_INVERT, cfg.ADAPTIVE_BLOCK_SIZE)
    cleaned = morphological_cleanup(
        binary, cfg.MORPH_OPEN_KERNEL, cfg.MORPH_CLOSE_KERNEL, cfg.MORPH_ITERATIONS
    )
    return {
        "original": resized,   # BGR image actually used for colour + annotation
        "grayscale": gray,
        "denoised": blurred,
        "enhanced": enhanced,
        "binary": cleaned,
        "was_resized": was_resized,
    }
