"""Annotated-image rendering.

Draws detected particle contours, IDs, predicted classes and (uncalibrated)
confidence values onto a copy of the analysed image. The result is returned to
the frontend as a base64 PNG data URI.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.schemas.analysis import ParticleResult

# BGR colours per morphology class (distinct, colour-blind friendly-ish set).
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "fiber": (60, 200, 255),      # amber/orange
    "fragment": (80, 80, 255),    # red
    "bead": (255, 170, 50),       # sky blue
    "other": (160, 160, 160),     # grey
}
DEFAULT_COLOR = (0, 255, 0)


def annotate_image(
    image_bgr: np.ndarray,
    particles: list[ParticleResult],
    contours: list[np.ndarray],
    draw_confidence: bool = True,
) -> np.ndarray:
    """Return a copy of the image with contour overlays and labels drawn.

    ``particles`` and ``contours`` must be index-aligned (both are produced by
    the same detection pass in the analysis pipeline).
    """
    annotated = image_bgr.copy()

    for particle, contour in zip(particles, contours):
        color = CLASS_COLORS.get(particle.particle_class, DEFAULT_COLOR)

        cv2.drawContours(annotated, [contour], -1, color, 2)

        x, y, width, height = cv2.boundingRect(contour)
        label = f"#{particle.id} {particle.particle_class}"
        if draw_confidence:
            label += f" {particle.confidence:.2f}"

        # Label background plate for readability.
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        plate_top = max(0, y - text_height - baseline - 4)
        cv2.rectangle(
            annotated,
            (x, plate_top),
            (x + text_width + 4, plate_top + text_height + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 2, plate_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    return annotated
