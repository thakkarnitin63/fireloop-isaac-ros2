"""Launch file for BT pick and plcae node with MoveIt Configs."""

import os
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def _moveit_params(moveit_config):
    """Extract all MoveIt params - same helper as bringup."""
    if hasattr(moveit_config, "to_dict"):
        return moveit_config.to_dict()

    params = {}
    for attr in [
        "robot_description",
        "robot_description_semantic",
        "robot_description_kinematics",
        "planning_pipelines",
        "trajectory_execution",
        "planning_scene_monitor_parameters",
        "moveit_cpp",
    ]:
        if hasattr(moveit_config, attr):
            v = getattr(moveit_config, attr)
            if isinstance(v, dict):
                params.update(v)
            else:
                try:
                    params.update(dict(v))
                except Exception:
                    pass

    return params

def generate_launch_description():
    moveit_config_pkg = get_package_share_directory("so101_moveit_config")

    moveit_config = (
        MoveItConfigsBuilder("so101_new_calib", package_name="so101_moveit_config")
        .moveit_cpp(
            file_path=os.path.join(moveit_config_pkg, "config", "moveit_cpp.yaml")
        )
        .to_moveit_configs()
    )

    moveit_params = _moveit_params(moveit_config)

    # Override robot_description with xacro (same as bringup)
    robot_description_content = ParameterValue(
        Command([
            "xacro ",
            PathJoinSubstitution([
                FindPackageShare("so101_moveit_config"),
                "config",
                "so101_new_calib.urdf.xacro",
            ]),
            " use_fake_hardware:=true",
            " use_sim_time:=true",
        ]),
        value_type=str,
    )
    moveit_params["robot_description"] = robot_description_content


    bt_node = Node(
        package = "so101_state_machine",
        executable = "bt_node",
        name = "bt_pick_place_node",
        output = "screen",
        parameters = [moveit_params, {"use_sim_time": True}],
    )

    perception_node = Node(
        package = "so101_perception",
        executable = "perception_node",
        name = "perception_node",
        output = "screen",
        parameters = [{"use_sim_time": True}], 
    )
    
    return LaunchDescription([
        bt_node,
        perception_node,
    ])


