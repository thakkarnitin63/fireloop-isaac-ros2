"""
Grasp pose estimation from a 3D point cloud of the detected object.


Design decisions:
- Expects points in base_link frame (Z-up, gravity-aligned)
- PCA on point cloud gives object orientation
- scipy.spatial.transform for all rotation math
- Grasp approach is top-down (negative Z in base_link)
- Pre-grasp offset configurable for safe approach
- Returns a full 6-DOF pose (position + quaternion)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GraspConfig:
    """Tunable parameters for grasp pose computation."""
    pre_grasp_offset: float = 0.05      # meters BEHIND the cup along approach
    grasp_depth_offset: float = 0.01    # fine adjustment toward cup center
    min_points: int = 10
    grasp_height_offset: float = 0.0    # vertical offset from cup center
    approach_height: float = 0.08       # NEW: how high above cup to approach


@dataclass
class GraspPose:
    """Complete grasp specification."""
    position: np.ndarray               # [x, y, z] grasp point in base_link frame
    orientation: np.ndarray             # quaternion [x, y, z, w]
    pre_grasp_position: np.ndarray     # approach point above object
    confidence: float
    jaw_reach: float = 0.05 

    def orientation_as_euler(self, degrees: bool = True) -> np.ndarray:
        """Convenience: get orientation as roll-pitch-yaw."""
        return Rotation.from_quat(self.orientation).as_euler('xyz', degrees=degrees)
    

from scipy.optimize import least_squares

def _fit_circle_2d(points_xy: np.ndarray):
    """
    Fit a circle to 2D points using least-squares.
    Returns (cx, cy, radius).
    Works great even with partial arcs (single camera view).
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    
    # Initial guess: centroid and mean distance
    cx0, cy0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - cx0)**2 + (y - cy0)**2))
    
    def residuals(params):
        cx, cy, r = params
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r
    
    result = least_squares(residuals, [cx0, cy0, r0])
    cx, cy, r = result.x
    return cx, cy, abs(r)


# def estimate_grasp(
#     points: np.ndarray,
#     config: GraspConfig = GraspConfig()
# ) -> Optional[GraspPose]:
#     """
#     Estimate a 6-DOF grasp pose from object point cloud in base_link frame.

#     Since points are in base_link (Z-up), top-down grasp means:
#     - Approach along -Z (downward)
#     - Gripper opening aligned with object's widest horizontal axis

#     Pipeline:
#         points -> centroid (position) -> PCA (orientation) ->
#         build grasp frame -> compute approach pose -> score

#     Args:
#         points: Nx3 array of 3D points in base_link frame
#         config: Grasp parameters

#     Returns:
#         GraspPose or None if estimation fails
#     """
#     if points is None or len(points) < config.min_points:
#         return None

#     # Step 1: Position — handle partial point cloud bias
#     # XY: use median instead of mean - more robust to one-sided view
#     # grasp_x = float(np.median(points[:, 0]))
#     # grasp_y = float(np.median(points[:, 1]))
#     # z_mid = float(np.median(points[:, 2]))
#     grasp_x = float(np.mean(points[:, 0]))
#     grasp_y = float(np.mean(points[:, 1]))
#     z_mid = float(np.mean(points[:, 2]))

#     # Approach direction: from robot base (0,0) toward cup, in XY plane
#     approach_dir = np.array([grasp_x, grasp_y, 0.0])
#     approach_norm = np.linalg.norm(approach_dir)
#     if approach_norm < 1e-6:
#         approach_dir = np.array([1.0, 0.0, 0.0])
#     else:
#         approach_dir /= approach_norm
    
#     # Orientation from PCA (still useful for gripper roll)
#     axes, eigenvalues = _compute_pca(points)
#     if axes is None:
#         return None    

#     # grasp_rotation = _build_grasp_rotation(axes)
#     # grasp_rotation = _ensure_valid_rotation(grasp_rotation)
#     grasp_rotation = _build_side_grasp_rotation(approach_dir)
#     # quaternion = Rotation.from_matrix(grasp_rotation).as_quat()

#     confidence = _compute_confidence(points, eigenvalues)

#     grasp_position = np.array([
#     grasp_x - (0.12 * approach_dir[0]),
#     grasp_y - (0.09 * approach_dir[1]) + 0.03,  # shift left (+Y)
#     z_mid + 0.1                                  # lower
# ])

#     # Pre-grasp: same XY, but high above
#     pre_grasp_position = np.array([
#         grasp_position[0],
#         grasp_position[1],
#         z_mid + 0.12
#     ])
#     # pre_grasp_position[2] += 0.0  # Lifts the gripper 2cm to clear the table surface
#     quaternion = np.array([0.0, 0.0, 0.0, 1.0])
#     grasp_rotation = np.eye(3)
#     return GraspPose(
#         position=grasp_position,
#         orientation=quaternion,
#         pre_grasp_position=pre_grasp_position,
#         rotation_matrix=grasp_rotation,
#         confidence=confidence,
#     )

def estimate_grasp(
    points: np.ndarray,
    config: GraspConfig = GraspConfig(),
    jaw_reach: float = 0.05
) -> Optional[GraspPose]:
    
    if points is None or len(points) < config.min_points:
        return None

    
    cup_cx, cup_cy, cup_radius = _fit_circle_2d(points[:, :2])
    z_mid = float(np.mean(points[:, 2]))
    
    # Approach direction: from robot base toward cup center
    approach_dir = np.array([cup_cx, cup_cy, 0.0])
    approach_norm = np.linalg.norm(approach_dir)
    if approach_norm < 1e-6:
        approach_dir = np.array([1.0, 0.0, 0.0])
    else:
        approach_dir /= approach_norm

    confidence = min(len(points) / 500.0, 1.0)

    # Where we want the JAW to be: at the cup surface
    jaw_target = np.array([
        cup_cx - cup_radius * approach_dir[0],  # cup edge
        cup_cy - cup_radius * approach_dir[1],
        z_mid + 0.11
    ])
    # Approximate gripper_link -> jaw distance along approach
    # This gets refined by TF after execution
    clearance = 0.02  # 2cm buffer so gripper body doesn't hit cup
    total_offset = jaw_reach + clearance


    # Where gripper_link needs to be so jaw reaches the cup
    grasp_position = np.array([
        jaw_target[0] - (jaw_reach + 0.02) * approach_dir[0],
        jaw_target[1] - (jaw_reach + 0.02) * approach_dir[1] +0.01,
        jaw_target[2]
    ])

    pre_grasp_position = np.array([
        jaw_target[0] - (total_offset + 0.07) * approach_dir[0],
        jaw_target[1] - (total_offset + 0.07) * approach_dir[1] + 0.01,
        jaw_target[2]
    ])

    quaternion = np.array([0.0, 0.0, 0.0, 1.0])


    print(f'[GraspEstimator] cup=({cup_cx:.3f}, {cup_cy:.3f}), '
          f'r={cup_radius:.3f}, z={z_mid:.3f}, '
          f'jaw_target=({jaw_target[0]:.3f}, {jaw_target[1]:.3f}, {jaw_target[2]:.3f}), '
          f'gripper_target=({grasp_position[0]:.3f}, {grasp_position[1]:.3f}, {grasp_position[2]:.3f})', flush=True)

    return GraspPose(
        position=grasp_position,
        orientation=quaternion,
        pre_grasp_position=pre_grasp_position,
        confidence=confidence,
        jaw_reach=jaw_reach
    )


def _compute_pca(
    points: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    PCA using scipy's eigh (numerically stable for symmetric matrices).

    Returns:
        axes: 3x3, each row is a principal axis (largest variance first)
        eigenvalues: variance along each axis (descending)
    """
    if len(points) < 3:
        return None, None

    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered, rowvar=False)

    # scipy.linalg.eigh is more stable than np.linalg.eigh
    eigenvalues, eigenvectors = eigh(cov)

    # Flip to descending order (largest variance first)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    return eigenvectors.T, eigenvalues


def _build_grasp_rotation(axes: np.ndarray) -> np.ndarray:
    """
    Build a rotation matrix for top-down grasping in base_link frame.

    In base_link frame (Z-up):
    - Grasp Z axis: points DOWN (-Z in base_link) — approach direction
    - Grasp X axis: aligned with object's widest HORIZONTAL axis (gripper opening)
    - Grasp Y axis: completes right-hand frame

    For a cup on a table:
    - PCA axis most aligned with Z is the cup's vertical axis
    - The two axes most in the XY plane are the cup's cross-section
    """
    # Find which PCA axis is most vertical (most aligned with Z)
    z_world = np.array([0.0, 0.0, 1.0])
    z_alignments = np.abs(axes @ z_world)
    vertical_idx = np.argmax(z_alignments)

    # Approach direction: straight down in base_link
    approach = np.array([0.0, 0.0, -1.0])

    # Gripper opening: widest horizontal PCA axis
    # Pick the largest-variance axis that ISN'T the vertical one
    horizontal_indices = [i for i in range(3) if i != vertical_idx]
    opening = axes[horizontal_indices[0]].copy()

    # Project opening onto XY plane (remove any Z component)
    opening[2] = 0.0
    opening_norm = np.linalg.norm(opening)
    if opening_norm < 1e-6:
        # Fallback: use X axis if opening is degenerate
        opening = np.array([1.0, 0.0, 0.0])
    else:
        opening /= opening_norm

    # Side axis completes the right-hand frame
    side = np.cross(approach, opening)
    side_norm = np.linalg.norm(side)
    if side_norm < 1e-6:
        side = np.array([0.0, 1.0, 0.0])
    else:
        side /= side_norm

    # Recompute opening for perfect orthogonality
    opening = np.cross(side, approach)
    opening /= np.linalg.norm(opening)

    return np.column_stack([opening, side, approach])

def _build_side_grasp_rotation(approach_dir: np.ndarray) -> np.ndarray:
    """
    Builds a rotation matrix for a horizontal (side) grasp.
    
    In base_link frame:
    - Grasp Z (Approach): Points from robot toward cup (approach_dir)
    - Grasp Y (Up): Points straight up [0, 0, 1]
    - Grasp X (Side): Completes the right-hand frame
    """
    # 1. The Approach (Z-axis of the gripper)
    # We use the XY direction we calculated from the robot to the cup
    grasp_z = approach_dir / np.linalg.norm(approach_dir)
    
    # 2. The 'Up' direction (Y-axis of the gripper)
    # Keeping the gripper level with the horizon
    grasp_y = np.array([0.0, 0.0, 1.0])
    
    # 3. The 'Side' direction (X-axis of the gripper)
    # Perpendicular to both approach and up
    grasp_x = np.cross(grasp_y, grasp_z)
    grasp_x /= np.linalg.norm(grasp_x)
    
    # Re-calculate Y to ensure perfect orthogonality
    grasp_y = np.cross(grasp_z, grasp_x)
    
    # Construct the 3x3 rotation matrix (columns are X, Y, Z)
    return np.column_stack([grasp_x, grasp_y, grasp_z])


def _ensure_valid_rotation(rot: np.ndarray) -> np.ndarray:
    """
    Project matrix to nearest valid rotation using SVD.
    Ensures det(R) = +1 and R^T @ R = I.
    """
    u, _, vt = np.linalg.svd(rot)
    corrected = u @ vt

    if np.linalg.det(corrected) < 0:
        u[:, -1] *= -1
        corrected = u @ vt

    return corrected


def _compute_confidence(
    points: np.ndarray,
    eigenvalues: np.ndarray
) -> float:
    """
    Score grasp quality from point cloud characteristics.

    Factors:
    - Point density: more points = better coverage
    - Shape: eigenvalue ratios indicate cup-like geometry
    """
    density_score = min(len(points) / 500.0, 1.0)

    shape_score = 0.5
    if eigenvalues[0] > 1e-10:
        ratio = eigenvalues[2] / eigenvalues[0]
        if 0.05 < ratio < 0.6:
            shape_score = 1.0
        elif ratio < 0.05:
            shape_score = 0.3

    return 0.5 * density_score + 0.5 * shape_score




def estimate_rim_grasp(points: np.ndarray, config: GraspConfig = GraspConfig()) -> Optional[GraspPose]:
    if points is None or len(points) < config.min_points:
        return None

    # 1. Filter for RIM points (highest 10% of Z values)
    z_max = np.max(points[:, 2])
    rim_mask = points[:, 2] > (z_max - 0.02) # Top 2cm slice
    rim_points = points[rim_mask]

    if len(rim_points) < 5:
        return None

    # 2. Fit circle to the RIM points only
    rim_cx, rim_cy, rim_radius = _fit_circle_2d(rim_points[:, :2])
    
    # 3. Pick a grasp target on the rim closest to the robot (for safety)
    # Vector from cup center to robot base (0,0)
    vec_to_robot = np.array([-rim_cx, -rim_cy]) 
    vec_to_robot /= np.linalg.norm(vec_to_robot)
    
    # Grasp point is on the rim edge closest to robot
    grasp_x = rim_cx + (vec_to_robot[0] * rim_radius)
    grasp_y = rim_cy + (vec_to_robot[1] * rim_radius)
    grasp_z = z_max # Grasp at the very top

    grasp_position = np.array([grasp_x, grasp_y, grasp_z])

    # 4. Orientation: ALIGN WITH TANGENT
    # Approach vector (Z-axis of gripper) points DOWN (-Z world)
    # but maybe tilted slightly forward to clear the far rim
    approach_vec = np.array([0, 0, -1]) 
    
    # Gripper Close Axis (Y-axis or X-axis depending on gripper)
    # This must be perpendicular to the radius (Tangent to circle)
    # Radius vector is (grasp - center)
    radius_vec = np.array([grasp_x - rim_cx, grasp_y - rim_cy, 0])
    radius_vec /= np.linalg.norm(radius_vec)
    
    # Tangent is cross product of radius and vertical
    tangent_vec = np.cross(np.array([0,0,1]), radius_vec)
    
    # Build Rotation Matrix
    # X = Tangent (Jaws move along this line)
    # Z = Approach (Down)
    # Y = Normal (Pointing into cup center)
    x_axis = tangent_vec
    z_axis = approach_vec
    y_axis = np.cross(z_axis, x_axis) # Should point roughly to center
    
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

    # Pre-grasp is just 10cm above the rim
    pre_grasp_position = grasp_position + np.array([0, 0, 0.10])

    print(f'[RimGrasp] Rim Center: ({rim_cx:.3f}, {rim_cy:.3f}) Radius: {rim_radius:.3f}')
    
    return GraspPose(
        position=grasp_position,
        orientation=quaternion,
        pre_grasp_position=pre_grasp_position,
        rotation_matrix=rotation_matrix,
        confidence=1.0
    )
