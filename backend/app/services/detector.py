"""Candidate particle detection.

OpenCV contour analysis on the cleaned binary mask. IMPORTANT: every contour
is only a *candidate* particle - there is no evidence at this stage that it is
a microplastic. Classification (and even that is only morphological) happens
later, in the Random Forest stage.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings, settings


def detect_candidate_particles(
    binary_mask: np.ndarray, config: Settings | None = None
) -> list[np.ndarray]:
    """Find external contours whose area passes the configured noise filters.

    - Contours smaller than ``MIN_PARTICLE_AREA_PX`` are treated as noise.
    - Contours larger than ``MAX_PARTICLE_AREA_PX`` (if enabled) are ignored,
      e.g. large smears or an inverted background.
    - Results are sorted by area (largest first) for stable, readable output.
    """
    cfg = config or settings
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[np.ndarray] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < cfg.MIN_PARTICLE_AREA_PX:
            continue
        if cfg.MAX_PARTICLE_AREA_PX > 0 and area > cfg.MAX_PARTICLE_AREA_PX:
            continue
        candidates.append(contour)

    candidates.sort(key=cv2.contourArea, reverse=True)
    return candidates
