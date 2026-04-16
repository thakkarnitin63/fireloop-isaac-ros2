"""
Perception ROS 2 node - Orchestrator

Responsibilities:
- Subscribe to camera topics
- Expose /detect_cup service
- Orchestrate: segment -> validate -> pointcloud -> transform -> grasp
- Publish results (pose, debug image, RViz markers)
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import struct
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation

# Our pure modules — no ROS inside them
from so101_perception.color_segmenter import segment, SegmentationConfig
from so101_perception.shape_validator import validate, ShapeConfig
from so101_perception.depth_estimator import (
    create_pointcloud_from_mask, compute_centroid, CameraIntrinsics, DepthConfig
)
from so101_perception.grasp_estimator import estimate_grasp, GraspConfig

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # ROS params
        self.declare_parameter('target_color', 'red')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('confidence_threshold', 0.4)

        # Config for mods
        self.seg_config = SegmentationConfig()
        self.shape_config = ShapeConfig()
        self.depth_config = DepthConfig()
        self.grasp_config = GraspConfig()

        # State: Latest images (updated by subscribers)
        self._bridge = CvBridge()
        self._latest_rgb = None
        self._latest_depth = None
        self._intrinsics = None


        # TF listener
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Subscribers (sensor data)
        sensor_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history = HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            Image, '/camera/color/image_raw',
            self._rgb_callback, sensor_qos
        )
        self.create_subscription(
            Image, '/camera/depth/image_raw',
            self._depth_callback, sensor_qos
        )
        self.create_subscription(
            CameraInfo, '/camera/camera_info',
            self._info_callback, sensor_qos
        )

        
        
        # --- Service ---
        self.create_service(Trigger, '/detect_cup', self._detect_callback)

        # --- Publishers ---
        self._pose_pub = self.create_publisher(
            PoseStamped, '/detected_cup_pose', 10
        )
        self._debug_pub = self.create_publisher(
            Image, '/perception/debug_image', 10
        )
        self._marker_pub = self.create_publisher(
            Marker, '/perception/grasp_marker', 10
        )

        self._cloud_pub = self.create_publisher(
            PointCloud2, '/perception/pointcloud_base', 10
        )
        self._pre_pose_pub = self.create_publisher(
            PoseStamped, '/pre_grasp_pose', 10)

        self.get_logger().info('Perception node ready. Call /detect_cup to detect.')




    # ---------------------------------------------------------
    # Subscriber callbacks — just store latest data
    # ---------------------------------------------------------

    def _rgb_callback(self, msg: Image):
        self._latest_rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _depth_callback(self, msg: Image):
        self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_callback(self, msg: CameraInfo):
        if self._intrinsics is None:
            self._intrinsics = CameraIntrinsics(
                fx=msg.k[0], fy=msg.k[4],
                cx=msg.k[2], cy=msg.k[5],
                width=msg.width, height=msg.height
            )
            self.get_logger().info(
                f'Camera intrinsics: fx={self._intrinsics.fx:.1f}, '
                f'fy={self._intrinsics.fy:.1f}, '
                f'cx={self._intrinsics.cx:.1f}, cy={self._intrinsics.cy:.1f}'
            )


    # ---------------------------------------------------------
    # Service callback — the main pipeline
    # ---------------------------------------------------------

    def _detect_callback(self, request, response):
        """
        Full detection pipeline. Called on demand by BT or user.

        Flow:
            check data -> segment color -> validate shape ->
            create pointcloud (camera frame) -> transform to base_link ->
            estimate grasp -> publish results
        """
        color = self.get_parameter('target_color').value
        base_frame = self.get_parameter('base_frame').value
        camera_frame = self.get_parameter('camera_frame').value
        threshold = self.get_parameter('confidence_threshold').value

         # Step 0: Check we have data
        if self._latest_rgb is None or self._latest_depth is None:
            response.success = False
            response.message = 'No camera data received yet'
            self.get_logger().warn(response.message)
            return response
        
        if self._intrinsics is None:
            response.success = False
            response.message = 'No camera intrinsics received yet'
            self.get_logger().warn(response.message)
            return response
        
        # Step 1: Color segmentation
        mask, pixel_count = segment(
            self._latest_rgb, color, self.seg_config
        )

        if mask is None:
            response.success = False
            response.message = f'No {color} pixels detected (count: {pixel_count})'
            self._publish_debug_image(self._latest_depth, None, None)
            return response
        

        # Step 2: Shape validation
        candidates = validate(mask, self.shape_config)
        if not candidates:
            response.success = False
            response.message = f'{color} pixels found but no valid shapes'
            self._publish_debug_image(self._latest_rgb, mask, None)
            return response
        
        best = candidates[0]
        self.get_logger().info(
            f'Best candidate: centroid={best.centroid}, '
            f'area={best.area:.0f}, confidence={best.confidence:.2f}'
        )

        # Step 3: Create pointcloud in camera frame 
        points_camera = create_pointcloud_from_mask(
            self._latest_depth, mask, self._intrinsics, self.depth_config
        )

        if points_camera is None:
            response.success = False
            response.message = 'Failed to create point cloud from depth'
            self._publish_debug_image(self._latest_rgb, mask, best)
            return response
        # Step 3b: Alternative position from 2D centroid + single depth lookup
        # More robust than 3D point cloud centroid for partial views

        cx, cy = best.centroid
        depth_at_center = self._latest_depth[cy, cx]

        if np.isfinite(depth_at_center) and depth_at_center > 0.1:
            from so101_perception.depth_estimator import CameraIntrinsics
            # Back-project the 2D center to 3D
            z = float(depth_at_center)
            x = float((cx - self._intrinsics.cx) * z / self._intrinsics.fx)
            y = float((cy - self._intrinsics.cy) * z / self._intrinsics.fy)
            self.get_logger().info(
                f'2D centroid depth method: camera frame [{x:.3f}, {y:.3f}, {z:.3f}]'
            )
        
        # Step 4: Transform pointcloud to base_link frame
        points_base = self._transform_points(
            points_camera, camera_frame, base_frame
        )
        if points_base is None:
            response.success = False
            response.message = f'TF transform {camera_frame} -> {base_frame} failed'
            return response
        
        # Step 5: Estimate grasp pose in base_link frame



        try:
            tf_gripper = self._tf_buffer.lookup_transform(
                'base_link', 'gripper_link', rclpy.time.Time()
            )
            tf_jaw = self._tf_buffer.lookup_transform(
                'base_link', 'moving_jaw_so101_v1_link', rclpy.time.Time()
            )
            jaw_offset = np.array([
                tf_jaw.transform.translation.x - tf_gripper.transform.translation.x,
                tf_jaw.transform.translation.y - tf_gripper.transform.translation.y,
                tf_jaw.transform.translation.z - tf_gripper.transform.translation.z,
            ])
            jaw_reach = float(np.linalg.norm(jaw_offset))
            self.get_logger().info(f'TF jaw_reach: {jaw_reach:.4f}m')
        except TransformException as e:
            self.get_logger().warn(f'TF failed: {e}, using default')
            jaw_reach = 0.05

        # Then estimate grasp with TF-derived offset
        grasp = estimate_grasp(points_base, self.grasp_config, jaw_reach=jaw_reach)
        if grasp is None:
            response.success = False
            response.message = 'Grasp estimation failed - insufficient points'
            return response

        if grasp.confidence < threshold:
            response.success = False
            response.message = (
                f'Grasp confidence too low: {grasp.confidence:.2f} < {threshold}'
            )
            return response
        

        # Step 6: Publish results
        self._publish_pose(grasp, base_frame)
        self._publish_pre_grasp_pose(grasp, base_frame)
        self._publish_grasp_marker(grasp, base_frame)
        self._publish_debug_image(self._latest_rgb, mask, best)
        self._publish_pointcloud(points_base, base_frame)

        response.success = True
        response.message = (
            f'{color}_cup detected | confidence: {grasp.confidence:.2f} | '
            f'position: [{grasp.position[0]:.3f}, '
            f'{grasp.position[1]:.3f}, {grasp.position[2]:.3f}]'
        )
        self.get_logger().info(response.message)
        return response
    

    # ---------------------------------------------------------
    # TF transform — camera frame to base frame
    # ---------------------------------------------------------

    def _transform_points(
        self,
        points: np.ndarray,
        source_frame: str,
        target_frame: str
    ) -> np.ndarray:
        """
        Transform Nx3 point cloud from source to target frame using tf2.
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
        except TransformException as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return None

        # Extract rotation and translation
        t = transform.transform.translation
        r = transform.transform.rotation
        translation = np.array([t.x, t.y, t.z])
        rotation = Rotation.from_quat([r.x, r.y, r.z, r.w])

        # Apply transform: p_target = R @ p_source + t
        points_transformed = rotation.apply(points) + translation
        return points_transformed
    



    # ---------------------------------------------------------
    # Publishers — pose, debug image, RViz marker
    # ---------------------------------------------------------

    def _publish_pose(self, grasp, frame_id: str):
        """Publish detected grasp pose as PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.pose.position.x = float(grasp.position[0])
        msg.pose.position.y = float(grasp.position[1])
        msg.pose.position.z = float(grasp.position[2])

        msg.pose.orientation.x = float(grasp.orientation[0])
        msg.pose.orientation.y = float(grasp.orientation[1])
        msg.pose.orientation.z = float(grasp.orientation[2])
        msg.pose.orientation.w = float(grasp.orientation[3])

        self._pose_pub.publish(msg)

    def _publish_pre_grasp_pose(self, pre_grasp, frame_id: str):
        """Publish detected pre_grasp pose as PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.pose.position.x = float(pre_grasp.pre_grasp_position[0])
        msg.pose.position.y = float(pre_grasp.pre_grasp_position[1])
        msg.pose.position.z = float(pre_grasp.pre_grasp_position[2])

        msg.pose.orientation.x = float(pre_grasp.orientation[0])
        msg.pose.orientation.y = float(pre_grasp.orientation[1])
        msg.pose.orientation.z = float(pre_grasp.orientation[2])
        msg.pose.orientation.w = float(pre_grasp.orientation[3])

        self._pre_pose_pub.publish(msg)




    def _publish_grasp_marker(self, grasp, frame_id: str):
        """Publish grasp pose as an arrow marker in RViz."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.ns = 'grasp'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.position.x = float(grasp.position[0])
        marker.pose.position.y = float(grasp.position[1])
        marker.pose.position.z = float(grasp.position[2])

        marker.pose.orientation.x = float(grasp.orientation[0])
        marker.pose.orientation.y = float(grasp.orientation[1])
        marker.pose.orientation.z = float(grasp.orientation[2])
        marker.pose.orientation.w = float(grasp.orientation[3])

        marker.scale.x = 0.08   # arrow length
        marker.scale.y = 0.01   # arrow width
        marker.scale.z = 0.01   # arrow height

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime.sec = 10

        self._marker_pub.publish(msg=marker)

    def _publish_pointcloud(self, points: np.ndarray, frame_id: str):
        """
        Publish Nx3 numpy array as PointCloud2 in RViz.
        Useful for verifying TF transform is correct.
        """
        if points is None or len(points) == 0:
            return

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.point_step = 12       # 3 floats × 4 bytes
        msg.row_step = msg.point_step * len(points)
        msg.is_bigendian = False
        msg.is_dense = True

        # Pack points as binary float32
        msg.data = points.astype(np.float32).tobytes()

        self._cloud_pub.publish(msg)
        self.get_logger().info(f'Published {len(points)} points in {frame_id}')



    def _publish_debug_image(self, rgb, mask, candidate):
        """
        Publish annotated image showing detection results.
        Green contour + crosshair on detected object.
        """
        debug = rgb.copy()

        if mask is not None:
            # Draw all detected color pixels as semi-transparent overlay
            overlay = debug.copy()
            overlay[mask > 0] = [0, 255, 0]
            debug = cv2.addWeighted(debug, 0.7, overlay, 0.3, 0)

        if candidate is not None:
            # Draw bounding box
            x, y, w, h = candidate.bbox
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw centroid crosshair
            cx, cy = candidate.centroid
            cv2.drawMarker(
                debug, (cx, cy), (0, 0, 255),
                cv2.MARKER_CROSS, 20, 2
            )

            # Draw confidence text
            label = f'{candidate.confidence:.2f}'
            cv2.putText(
                debug, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        msg = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        self._debug_pub.publish(msg)
    


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    


