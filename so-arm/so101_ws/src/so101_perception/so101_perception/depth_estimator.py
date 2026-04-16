"""
Depth estimation module for 3D position recovery.
Pure math - takes pixels + depth + intrinsics, returns 3D points.

Design decisions:
- Back-projects 2D pixels to 3D using pinhole camera model
- Handles invalid depth values (zero, NaN, inf)
- Can process single points or entire masked regions
- Returns points in camera frame - TF transform is caller's job
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class CameraIntrinsics:
    """Camera parameters from CameraInfo message."""
    fx: float      # focal length x (pixels)
    fy: float      # focal length y (pixels)
    cx: float      # principal point x (pixels)
    cy: float      # principal point y (pixels)
    width: int     # image width
    height: int    # image height


@dataclass
class DepthConfig:
    """Tunable parameters for depth processing."""
    min_depth: float = 0.1        # meters - closer than this is noise
    max_depth: float = 2.0        # meters - farther than this is background
    depth_scale: float = 1.0      # multiplier if depth is in mm (set to 0.001)
    outlier_std_factor: float = 2.0   # reject points beyond N std devs from median
    min_valid_points: int = 10        # minimum points for a usable cloud


def create_pointcloud_from_mask(
    depth_image: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: DepthConfig = DepthConfig()
) -> Optional[np.ndarray]:
    """
    Convert masked depth pixels into a 3D point cloud.

    Pipeline:
        mask -> pixel coords -> depth lookup -> filter invalid ->
        reject outliers -> back-project -> Nx3 point cloud

    Args:
        depth_image: HxW depth image (float32, meters from Isaac Sim)
        mask: HxW binary mask (255 = object pixels)
        intrinsics: Camera parameters from CameraInfo
        config: Depth processing parameters

    Returns:
        Nx3 numpy array of [X, Y, Z] in camera frame, or None
    """
    if depth_image is None or mask is None:
        return None

    # Extract pixel coordinates where mask is active
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        return None

    # Lookup depth at those pixels, apply scale
    depths = depth_image[vs, us].astype(np.float64) * config.depth_scale

    # Filter invalid depths (zero, NaN, inf, out of range)
    valid = (
        np.isfinite(depths) &
        (depths > config.min_depth) &
        (depths < config.max_depth)
    )
    us, vs, depths = us[valid], vs[valid], depths[valid]

    if len(depths) < config.min_valid_points:
        return None

    # Reject outliers - a real object has consistent depth
    # Scattered depths mean noise or mixed foreground/background
    median_d = np.median(depths)
    std_d = np.std(depths)
    if std_d > 0:
        inliers = np.abs(depths - median_d) < config.outlier_std_factor * std_d
        us, vs, depths = us[inliers], vs[inliers], depths[inliers]

    if len(depths) < config.min_valid_points:
        return None

    # Vectorized back-projection (pinhole camera model)
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    points = np.column_stack([
        (us.astype(np.float64) - intrinsics.cx) * depths / intrinsics.fx,
        (vs.astype(np.float64) - intrinsics.cy) * depths / intrinsics.fy,
        depths
    ])   # (N,3)

    return points


def compute_centroid(points: np.ndarray) -> Optional[np.ndarray]:
    """
    Centroid of a point cloud - single numpy call.

    Args:
        points: Nx3 array

    Returns:
        [X, Y, Z] as numpy array, or None
    """
    if points is None or len(points) == 0:
        return None

    return np.mean(points, axis=0)


def compute_spread(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Spatial spread of the point cloud along each axis.
    Useful for sanity checking - a cup should be compact.

    Returns:
        (x_spread, y_spread, z_spread) in meters
    """
    if points is None or len(points) < 2:
        return None

    ranges = np.ptp(points, axis=0)  # peak-to-peak per axis
    return (float(ranges[0]), float(ranges[1]), float(ranges[2]))
