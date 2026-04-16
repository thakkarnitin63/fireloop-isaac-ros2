"""
Color segmentation module for object detection by color.
Pure vision - Takes an image + color name, returns a mask.

Design decisions:
- Any color name - auto-generates HSV detection ranges
- Uses matplotlib color names (supports hundreds of colors)
- Tolerance around center hue is configurable
- LAB validation as secondary check
- Fallback to predefined profiles for tricky colors like red
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict


@dataclass
class SegmentationConfig:
    """Global tuning knobs - not color-specific."""
    hue_tolerance: int = 20            # +/- degrees around target hue
    saturation_min: int = 35           # minimum saturation to count
    value_min: int = 15                # minimum brightness to count
    morph_kernel_size: int = 5         # cleanup kernel
    min_pixel_count: int = 200         # reject tiny detections
    lab_validate: bool = True          # use LAB as second check



def color_name_to_hsv_center(color_name: str) -> Tuple[int, int, int]:
    """
    Convert any color name to its HSV center value.
    Supports: 'red', 'blue', 'green', 'purple', 'orange',
    'yellow', 'cyan', 'pink', 'brown', etc.

    Uses matplotlib's named colors - hundreds supported.
    """
    try: 
        from matplotlib.colors import to_rgb
        rgb = to_rgb(color_name)     # returns (0-1, 0-1, 0-1)
    except (ImportError, ValueError):
        # Fallback: basic color lookup if matplotlib not available
        basic_colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255),
            'pink': (255, 105, 180),
            'brown': (139, 69, 19),
        }
        color_lower = color_name.lower()
        if color_lower not in basic_colors:
            raise ValueError(f"Unknown color: {color_name}")
        r, g, b = basic_colors[color_lower]
        rgb = (r / 255.0, g / 255.0, b / 255.0)

    # Convert to OpenCV HSV (H: 0-180, S: 0-255, V: 0-255)
    rgb_uint8 = np.array([[[
    int(rgb[0] * 255),
    int(rgb[1] * 255),
    int(rgb[2] * 255)
    ]]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    return int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])


def build_hsv_ranges(
    color_name: str,
    config: SegmentationConfig
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build HSV threshold ranges for any color.
    Handles red's hue wraparound automatically.
    """
    h_center, _, _ = color_name_to_hsv_center(color_name)
    tol = config.hue_tolerance

    h_low = h_center - tol
    h_high = h_center + tol

    # Handle hue wraparound (0-180 in OpenCV)
    if h_low < 0:
        # Color wraps around - need two ranges (e.g., red)
        return [
            (
                np.array([0, config.saturation_min, config.value_min]),
                np.array([h_high, 255, 255])
            ),
            (
                np.array([180 + h_low, config.saturation_min, config.value_min]),
                np.array([180, 255, 255])
            ),
        ]
    elif h_high > 180:
        # Also wraps around the other direction
        return [
            (
                np.array([h_low, config.saturation_min, config.value_min]),
                np.array([180, 255, 255])
            ),
            (
                np.array([0, config.saturation_min, config.value_min]),
                np.array([h_high - 180, 255, 255])
            ),
        ]
    else:
        # Normal case - single range
        return [
            (
                np.array([h_low, config.saturation_min, config.value_min]),
                np.array([h_high, 255, 255])
            ),
        ]
    
def segment(
        image_bgr: np.ndarray,
        color_name: str,
        config: SegmentationConfig = SegmentationConfig()
    ) -> Tuple[Optional[np.ndarray], int]:
    
    if image_bgr is None or image_bgr.size == 0:
        return None, 0
    
    # --- IMPROVEMENT 1: Bilateral Filter ---
    # Smoothes the object surface but keeps the cup/table edge sharp
    smoothed = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    
    # --- CLAHE for Lighting Robustness ---
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv_equalized = cv2.merge([h, s, v])

    # Step 1: HSV thresholding
    ranges = build_hsv_ranges(color_name, config)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges: 
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_equalized, lower, upper))

    # Step 2: LAB validation
    if config.lab_validate:
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        _, a_ch, b_ch = cv2.split(lab)
        h_center, _, _ = color_name_to_hsv_center(color_name)
        lab_mask = _build_lab_mask(a_ch, b_ch, h_center)
        mask = cv2.bitwise_and(mask, lab_mask)

    # Step 3: Morphological Cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

    # --- IMPROVEMENT 2: Convex Hull Geometry ---
    # This fills in highlights on the rim and gaps in the body
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (the cup)
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) > config.min_pixel_count:
            # Create a clean mask from the convex hull
            hull = cv2.convexHull(largest_cnt)
            refined_mask = np.zeros_like(mask)
            cv2.drawContours(refined_mask, [hull], -1, 255, thickness=-1)
            mask = refined_mask

    pixel_count = cv2.countNonZero(mask)
    if pixel_count < config.min_pixel_count:
        return None, pixel_count

    return mask, pixel_count



def _build_lab_mask(
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    hue_center: int
) -> np.ndarray:
    """
    Build a LAB-based validation mask based on the target hue.

    The idea: HSV tells us the hue, but LAB tells us the
    perceptual color axis. Using both reduces false positives.

    Hue mapping to LAB:
    (-) here is approx
    - Red (H  -0/180): high A channel
    - Green (H -60): low A channel
    - Blue (H -120): low B channel
    - Yellow (H -30): high B channel
    - Others: skip LAB validation (return all-white mask)
    """
    mask = np.ones(a_channel.shape, dtype=np.uint8) * 255

    if hue_center <= 10 or hue_center >= 170:
        # Red - A channel should be high
        _, mask = cv2.threshold(a_channel, 128, 255, cv2.THRESH_BINARY)
    elif 35 <= hue_center <= 85:
        # Green - A channel should be low
        _, mask = cv2.threshold(a_channel, 120, 255, cv2.THRESH_BINARY_INV)
    elif 100 <= hue_center <= 130:
        # Blue - B channel should be low
        _, mask = cv2.threshold(b_channel, 120, 255, cv2.THRESH_BINARY_INV)
    elif 20 <= hue_center <= 35:
        # Yellow/Orange - B channel should be high
        _, mask = cv2.threshold(b_channel, 140, 255, cv2.THRESH_BINARY)
    # For other hues, LAB mask is all-white (no filtering)

    return mask




