#!/usr/bin/env python3
"""
Behavior Tree node for SO101 pick-and-place.

Execution flow:
1. Setup and open gripper
2. Detect cup -> fallback to recovery on failure
3. Move to grasp -> close gripper
4. Attach (Sim) -> Move to Box -> Detach (Sim)
5. Release
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import py_trees
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped
from std_srvs.srv import Trigger

from moveit.planning import MoveItPy
from so101_planning.motion_planner import MotionPlanner

# -------------------------
# Action Leaves
# -------------------------

class OpenGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node

    def update(self) -> py_trees.common.Status:
        success, msg = self.node.planner.plan_and_execute_gripper('open')
        if success:
            return py_trees.common.Status.SUCCESS
        self.node.get_logger().error(f'[{self.name}] Failed: {msg}')
        return py_trees.common.Status.FAILURE
    
class CloseGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node

    def update(self) -> py_trees.common.Status:
        success, msg = self.node.planner.plan_and_execute_gripper('close')
        if success:
            return py_trees.common.Status.SUCCESS
        self.node.get_logger().error(f'[{self.name}] Failed: {msg}')
        return py_trees.common.Status.FAILURE
    
class DetectCup(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node
        self._client = self.node.create_client(Trigger, '/detect_cup')
        self._future = None
        self._start_time = None

    def initialise(self):
        self._start_time = time.monotonic()
        if self._client.wait_for_service(timeout_sec=2.0):
            self._future = self._client.call_async(Trigger.Request())
        else:
            self.node.get_logger().error('[DetectCup] Service unavailable')

    def update(self) -> py_trees.common.Status:
        if time.monotonic() - self._start_time > 10.0:
            self.node.get_logger().error('[DetectCup] Service timeout')
            return py_trees.common.Status.FAILURE

        if self._future and self._future.done():
            result = self._future.result()
            if result.success:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE
            
        return py_trees.common.Status.RUNNING
    
class MoveToPreGrasp(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node

    def update(self) -> py_trees.common.Status:
        if not self.node.pre_grasp_pose:
            self.node.get_logger().error('[MoveToPreGrasp] Missing pose data')
            return py_trees.common.Status.FAILURE

        success, msg = self.node.planner.plan_and_execute_pose(self.node.pre_grasp_pose.pose)
        if success:
            return py_trees.common.Status.SUCCESS
        
        self.node.get_logger().error(f'[MoveToPreGrasp] {msg}')
        return py_trees.common.Status.FAILURE

class MoveToGrasp(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node

    def update(self) -> py_trees.common.Status:
        if not self.node.detected_pose:
            return py_trees.common.Status.FAILURE

        target_pose = self.node.detected_pose.pose
        
        # Try Cartesian first, fallback to standard OMPL if kinematics complain
        success, msg = self.node.planner.plan_and_execute_cartesian(target_pose)
        if not success:
            self.node.get_logger().warn('[MoveToGrasp] Cartesian failed, switching to OMPL')
            success, msg = self.node.planner.plan_and_execute_pose(target_pose)

        if success:
            return py_trees.common.Status.SUCCESS
            
        self.node.get_logger().error(f'[MoveToGrasp] Planning failed: {msg}')
        return py_trees.common.Status.FAILURE

class MoveToBoxPosition(py_trees.behaviour.Behaviour):
    """Lifts up, translates over the bin, and lowers. Uses pure global coords."""
    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node
        self._phase = 0

    def initialise(self):
        self._phase = 0

    def update(self) -> py_trees.common.Status:
        try:
            # Grab target and current TFs
            bin_tf = self.node.tf_buffer.lookup_transform('base_link', 'Bin_B02_01', rclpy.time.Time())
            gripper_tf = self.node.tf_buffer.lookup_transform('base_link', 'gripper_link', rclpy.time.Time())
            
            try:
                jaw_tf = self.node.tf_buffer.lookup_transform('base_link', 'moving_jaw_so101_v1_link', rclpy.time.Time())
            except Exception:
                # Wait for TF tree to catch up
                return py_trees.common.Status.RUNNING

            # Calculate global offset to center the jaw over the bin
            delta_x = bin_tf.transform.translation.x - jaw_tf.transform.translation.x
            delta_y = bin_tf.transform.translation.y - jaw_tf.transform.translation.y

            # Manual offsets for Isaac Sim physics drift (adjust if cup clips the edge)
            tune_x = -0.075
            tune_y = 0.07

            target_x = gripper_tf.transform.translation.x + delta_x + tune_x
            target_y = gripper_tf.transform.translation.y + delta_y + tune_y
            
            safe_z = bin_tf.transform.translation.z + 0.26
            drop_z = bin_tf.transform.translation.z + 0.19

            pose = Pose()
            pose.orientation.w = 1.0

            if self._phase == 0:
                # Phase 0: Straight Z-lift
                pose.position.x = gripper_tf.transform.translation.x
                pose.position.y = gripper_tf.transform.translation.y
                pose.position.z = safe_z
                
                if self.node.planner.plan_and_execute_pose(pose)[0]:
                    self._phase = 1
                    return py_trees.common.Status.RUNNING
                return py_trees.common.Status.FAILURE

            elif self._phase == 1:
                # Phase 1: Translate XY
                pose.position.x = target_x
                pose.position.y = target_y
                pose.position.z = safe_z
                
                if self.node.planner.plan_and_execute_pose(pose)[0]:
                    self._phase = 2
                    return py_trees.common.Status.RUNNING
                return py_trees.common.Status.FAILURE

            elif self._phase == 2:
                # Phase 2: Lower into bin
                pose.position.x = target_x
                pose.position.y = target_y
                pose.position.z = drop_z
                
                if self.node.planner.plan_and_execute_pose(pose)[0]:
                    return py_trees.common.Status.SUCCESS
                return py_trees.common.Status.FAILURE

        except Exception as e:
            self.node.get_logger().warn(f'[MoveToBox] TF lookup failed: {e}')
            return py_trees.common.Status.RUNNING

class AttachDetachCube(py_trees.behaviour.Behaviour):
    def __init__(self, name, node, topic_name, attach, delay_sec=1.0):
        super().__init__(name)
        self.node = node
        self.topic_name = topic_name
        self.attach = attach
        self.delay_sec = delay_sec
        self.pub = self.node.create_publisher(Bool, topic_name, 10)

    def initialise(self):
        self._start_time = time.monotonic()
        self._publish_count = 0

    def update(self):
        msg = Bool()
        msg.data = self.attach
        self.pub.publish(msg)
        self._publish_count += 1

        if (time.monotonic() - self._start_time) >= self.delay_sec:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class RecoveryMoveUp(py_trees.behaviour.Behaviour):
    """Simple recovery: lift Z to clear any collisions."""
    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node

    def update(self) -> py_trees.common.Status:
        try:
            tf = self.node.tf_buffer.lookup_transform('base_link', 'gripper_link', rclpy.time.Time())
            pose = Pose()
            pose.position.x = tf.transform.translation.x
            pose.position.y = tf.transform.translation.y
            pose.position.z = tf.transform.translation.z + 0.15 # Retreat upwards
            pose.orientation.w = 1.0

            success, msg = self.node.planner.plan_and_execute_pose(pose)
            return py_trees.common.Status.SUCCESS if success else py_trees.common.Status.FAILURE
        except Exception:
            return py_trees.common.Status.FAILURE

# -------------------------
# Tree Assembly
# -------------------------
def create_tree(node: Node):
    STEP_RETRIES = 2
    ATTACH_TOPIC = "/isaac_attach_cube"
    EXECUTION_TIMEOUT = 15.0 # Max seconds allowed for a motion plan execution

    main_seq = py_trees.composites.Sequence(name="PickAndPlace", memory=True)

    # 1. Primary Grabbing Sequence
    grabbing_seq = py_trees.composites.Sequence(name='GrabbingSequence', memory=True)
    grabbing_seq.add_children([
        DetectCup('Detect', node),
        MoveToPreGrasp('PreGrasp', node),
        # Timeout applied here to catch IK/Motion hangs
        py_trees.decorators.Timeout(
            name="GraspTimeout",
            child=MoveToGrasp('Grasp', node),
            duration=EXECUTION_TIMEOUT
        ),
        
        AttachDetachCube('AttachCube', node, ATTACH_TOPIC, attach=True, delay_sec=2.0),
        # THEN close the physical jaws
        CloseGripper('CloseGripper', node)
    ])

    # 2. Recovery Sequence (Executes if Grabbing fails/times out)
    recovery_seq = py_trees.composites.Sequence(name='RecoverySequence', memory=True)
    recovery_seq.add_children([
        OpenGripper('RecoveryOpen', node),
        RecoveryMoveUp('RecoveryLift', node)
    ])

    # 3. Grasp Selector (Fallback mechanism)
    grasp_fallback = py_trees.composites.Selector(name="GraspWithRecovery", memory=False)
    grasp_fallback.add_children([
        py_trees.decorators.Retry('RetryGrabbing', grabbing_seq, STEP_RETRIES),
        recovery_seq
    ])

    # Build the main pipeline
    main_seq.add_children([
        OpenGripper('InitOpen', node),
        
        # This handles moving, attaching, and closing (or recovering if it fails)
        grasp_fallback, 
        
        py_trees.decorators.Timeout(
            name="MoveBoxTimeout",
            child=MoveToBoxPosition('MoveToBox', node),
            duration=EXECUTION_TIMEOUT * 2 # Box move has 3 phases, needs more time
        ),
        
        # Detach and release
        AttachDetachCube('DetachCube', node, ATTACH_TOPIC, attach=False, delay_sec=2.0),
        OpenGripper('FinalRelease', node)
    ])

    return py_trees.decorators.OneShot(
        name='RunOnce',
        child=main_seq,
        policy=py_trees.common.OneShotPolicy.ON_COMPLETION
    )

# -------------------------
# Node Definition
# -------------------------
class BTNode(Node):
    def __init__(self):
        super().__init__("bt_pick_place_node")
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('Initializing MoveItPy...')
        self._moveit = MoveItPy(node_name='moveit_bt')
        self.planner = MotionPlanner(self._moveit)

        self.detected_pose = None
        self.pre_grasp_pose = None
        
        self.create_subscription(PoseStamped, '/detected_cup_pose', self._pose_callback, 10)
        self.create_subscription(PoseStamped, '/pre_grasp_pose', self._pre_pose_callback, 10)

        self.tree = py_trees.trees.BehaviourTree(create_tree(self))
        # Note: MoveItPy execution can be blocking. The BT tick handles state updates 
        # around those blocking calls. 
        self.timer = self.create_timer(0.1, self._tick)

    def _pose_callback(self, msg: PoseStamped):
        self.detected_pose = msg

    def _pre_pose_callback(self, msg: PoseStamped):
        self.pre_grasp_pose = msg

    def _tick(self):
        self.tree.tick()

def main():
    rclpy.init()
    node = BTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()