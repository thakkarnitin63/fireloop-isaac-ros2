"""
Motion planner helper using official MoveItPy.
Handles planning only — execution goes through controller action servers.

NOT a ROS node. Initialized once, passed around.
"""

from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
import numpy as np
from typing import Optional, Tuple
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped


class MotionPlanner:
    """Wras MoveItPy for clean planning API."""

    def __init__(self, moveit: MoveItPy):
        self._moveit = moveit
        self._arm = moveit.get_planning_component('arm')
        self._gripper = moveit.get_planning_component('gripper')
        self._model = moveit.get_robot_model()
        self._scene = moveit.get_planning_scene_monitor()


        # --- CORRECT DEBUG PRINTS ---
        print("\n=== ROBOT SUMMARY ===", flush=True)
        
        # Explicitly ask the model for the 'arm' and 'gripper' groups
        arm_group = self._model.get_joint_model_group("arm")
        gripper_group = self._model.get_joint_model_group("gripper")
        
        print("\n=== GROUP LINKS ===", flush=True)
        print(f"Arm Links: {arm_group.link_model_names}", flush=True)
        print(f"Gripper Links: {gripper_group.link_model_names}", flush=True)
        print("=======================\n", flush=True)
        

    def plan_to_pose(self, pose: Pose) -> Optional[object]:
        """Plan arm to target pose. Returns trajectory or None."""
        self._arm.set_start_state_to_current_state()

        robot_state = self._arm.get_start_state()
        success = robot_state.set_from_ik(
            'arm', pose, 'gripper_frame_link'
        )
        if not success:
            return None
        
        self._arm.set_goal_state(robot_state=robot_state)
        result = self._arm.plan()
        return result.trajectory if result else None
    

    def plan_to_named(self, target_name: str) -> Optional[object]:
        """Plan arm to named SRDF state. Returns trajectory or None"""
        self._arm.set_start_state_to_current_state()
        self._arm.set_goal_state(configuration_name=target_name)
        result = self._arm.plan()
        return result.trajectory if result else None
    
    def plan_gripper(self, target_name: str) -> Optional[object]:
        """Plan gripper to named state ('open' or 'close')."""
        self._gripper.set_start_state_to_current_state()
        self._gripper.set_goal_state(configuration_name=target_name)
        result = self._gripper.plan()
        return result.trajectory if result else None
    
    def execute(self, trajectory) -> bool:
        """Execute a planned trajectory."""
        if trajectory is None:
            return False
        return self._moveit.execute(trajectory, controllers=[])
    
    def plan_and_execute_pose(self, pose: Pose) -> Tuple[bool, str]:
        """Convenience: plan + execute to pose."""
        traj = self.plan_to_pose(pose)
        if traj is None:
            return False, 'Planning failed - IK or planner error'
        success = self.execute(traj)
        return success, 'Execution succeeded' if success else 'Execution failed'
    

    def plan_and_execute_cartesian(self, target_pose: Pose) -> Tuple[bool, str]:
        """Move in a straight line to a target pose using Pilz Industrial Planner."""
        
        # 1. Safeguard: If bt_node passes a list, extract the final pose
        if isinstance(target_pose, list):
            target_pose = target_pose[-1]

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link" 
        pose_stamped.pose = target_pose

        self._arm.set_start_state_to_current_state()
        
        self._arm.set_goal_state(
            pose_stamped_msg=pose_stamped, 
            pose_link="gripper_frame_link"
        )
        
        # --- THE FIX ---
        # Initialize with the MoveItPy instance (self._moveit) and the pipeline namespace
        plan_params = PlanRequestParameters(self._moveit, "pilz_industrial_motion_planner")
        plan_params.planner_id = "LIN"
        plan_params.planning_pipeline = "pilz_industrial_motion_planner"
        
        # Pass the parameters to the planner
        result = self._arm.plan(single_plan_parameters=plan_params)
        
        if result and result.trajectory:
            success = self.execute(result.trajectory)
            return success, "Cartesian execution succeeded" if success else "Cartesian execution failed"
            
        return False, "Cartesian planning failed"
    

    
    def plan_and_execute_named(self, name: str) -> Tuple[bool, str]:
        """Convenience: plan + execute to named state."""
        traj = self.plan_to_named(name)
        if traj is None:
            return False, f'Planning to {name} failed'
        success = self.execute(traj)
        return success, f'{name} reached' if success else f'Execution to {name} failed'
    

    def plan_and_execute_gripper(self, name: str) -> Tuple[bool, str]:
        """Convenience: plan + execute gripper."""
        traj = self.plan_gripper(name)
        if traj is None:
            return False, f'Gripper planning to {name} failed'
        success = self.execute(traj)
        return success, f'Gripper {name}' if success else f'Gripper {name} failed'


    def add_cup_collision(self, cup_x, cup_y, cup_z, radius=0.04, height=0.10):
        """Add cup as collision cylinder so planner avoids it."""
        planning_scene = self._moveit.get_planning_scene_monitor()
        
        with planning_scene.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.id = "red_cup"
            collision_object.header.frame_id = "base_link"
            
            # Cup as a cylinder
            cylinder = SolidPrimitive()
            cylinder.type = SolidPrimitive.CYLINDER
            cylinder.dimensions = [height, radius]  # [height, radius]
            
            cup_pose = Pose()
            cup_pose.position.x = cup_x
            cup_pose.position.y = cup_y
            cup_pose.position.z = cup_z + (height / 2.0)  # center of cylinder
            cup_pose.orientation.w = 1.0
            
            collision_object.primitives.append(cylinder)
            collision_object.primitive_poses.append(cup_pose)
            collision_object.operation = CollisionObject.ADD
            
            scene.apply_collision_object(collision_object)

    def add_table_collision(self, table_z=0.0):
        """Add table as collision box."""
        planning_scene = self._moveit.get_planning_scene_monitor()
        
        with planning_scene.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.id = "table"
            collision_object.header.frame_id = "base_link"
            
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [1.0, 1.0, 0.02]  # wide, deep, thin
            
            table_pose = Pose()
            table_pose.position.x = 0.3
            table_pose.position.y = 0.0
            table_pose.position.z = table_z - 0.01
            table_pose.orientation.w = 1.0
            
            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(table_pose)
            collision_object.operation = CollisionObject.ADD
            
            scene.apply_collision_object(collision_object)

    def remove_cup_collision(self):
        """Remove cup collision before final approach."""
        planning_scene = self._moveit.get_planning_scene_monitor()
        
        with planning_scene.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.id = "red_cup"
            collision_object.operation = CollisionObject.REMOVE
            scene.apply_collision_object(collision_object)

