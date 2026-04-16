"""
Shape validation module for detected color regions.
Pure geometry - takes contours, scores them as object candidates.

Each candidate gets a confidence score based on:
- Circularity (how round is it)
- Area (is it a plausible object size)
- Solidity (is it a solid object, not scattered pixels)
- Aspect ratio (is it reasonably proportioned)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class ShapeConfig:
    """Tunable parameters for shape validation."""
    min_area: int = 300              # minimum contour area in pixels
    max_area: int = 50000            # maximum contour area in pixels
    min_circularity: float = 0.3     # 1.0 = perfect circle
    min_solidity: float = 0.6        # contour area / convex hull area
    max_aspect_ratio: float = 3.0    # width/height or height/width

    # Weights for final confidence score
    weight_circularity: float = 0.3
    weight_area: float = 0.2
    weight_solidity: float = 0.3
    weight_aspect: float = 0.2


@dataclass
class Candidate:
    """A detected object candidate with its properties."""
    contour: np.ndarray
    centroid: Tuple[int, int]        # (u, v) pixel coordinates
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    circularity: float
    solidity: float
    aspect_ratio: float
    confidence: float


def validate(
    mask: np.ndarray,
    config: ShapeConfig = ShapeConfig()
    ) -> List[Candidate]:
    """
    Find and score object candidates in a binary mask.

    Args:
        mask: Binary mask from color segmentation
        config: Shape validation parameters

    Returns:
        List of candidates sorted by confidence (highest first)
    """
    if mask is None:
        return []
    
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []
    for contour in contours:
        candidate = _score_contour(contour, config)
        if candidate is not None:
            candidates.append(candidate)

    # Sort by confidence, best first
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _score_contour(
    contour: np.ndarray, 
    config: ShapeConfig
    ) -> Optional[Candidate]:
    """
    Score a single contour. Returns None if its fails basic checks.
    """
    area = cv2.contourArea(contour)
    if area < config.min_area or area > config.max_area:
        return None
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    
    # --- Compute shape metrics ---

    # Circularity: 4π × area / perimeter²  (1.0 = perfect circle)
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    if circularity < config.min_circularity:
        return None

    # Solidity: contour area / convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < config.min_solidity:
        return None

    # Aspect ratio: of bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
    if aspect_ratio > config.max_aspect_ratio:
        return None

    # Centroid from moments
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return None
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # --- Compute confidence score ---
    # Each metric normalized to 0-1, then weighted

    circ_score = min(circularity / 0.8, 1.0)       # 0.8+ circularity = perfect score
    area_score = _area_score(area, config)
    solid_score = min(solidity / 0.9, 1.0)          # 0.9+ solidity = perfect score
    aspect_score = 1.0 - min((aspect_ratio - 1.0) / (config.max_aspect_ratio - 1.0), 1.0)

    confidence = (
        config.weight_circularity * circ_score +
        config.weight_area * area_score +
        config.weight_solidity * solid_score +
        config.weight_aspect * aspect_score
    )

    return Candidate(
        contour=contour,
        centroid=(cx, cy),
        bbox=(x, y, w, h),
        area=area,
        circularity=circularity,
        solidity=solidity,
        aspect_ratio=aspect_ratio,
        confidence=confidence,
    )


def _area_score(area: float, config: ShapeConfig) -> float:
    """
    Score area on a bell curve — too small or too large is bad.
    Ideal area is the midpoint of the allowed range.
    """
    ideal = (config.min_area + config.max_area) / 2.0
    distance = abs(area - ideal) / ideal
    return max(0.0, 1.0 - distance)






    





