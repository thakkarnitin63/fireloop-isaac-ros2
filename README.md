# SO101 Pick-and-Place: Perception + MoveIt + Behavior Tree

A perception-driven pick-and-place pipeline for the **SO101 6-DOF arm**, built on **ROS 2 Jazzy**, **Isaac Sim 5.1.0**, and **MoveIt 2**. The system detects a red cup with an RGB-D camera, plans a grasp using least-squares circle fitting, and transports the cup to a bin under the control of a `py_trees` Behavior Tree with retry/recovery logic.

> **Demo:** Robot detects the red cup (ignores the green one), plans a Cartesian approach, picks it up, moves it to the bin, and releases.

---

## Prerequisites

| Component | Required | Preferred |
|-----------|----------|-----------|
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |
| ROS 2 | Humble or newer | Jazzy |
| Isaac Sim | 4.x+ | 5.1.0 |
| MoveIt 2 | matching ROS 2 distro | matching ROS 2 distro |
| Python deps | `opencv-python`, `numpy`, `scipy`, `matplotlib`, `py_trees`, `cv_bridge` | |

Make sure your ROS 2 environment is sourced **before building**:

```bash
source /opt/ros/jazzy/setup.bash
```

Install Python dependencies:

```bash
pip install opencv-python numpy scipy matplotlib py-trees
```

---

## Workspace Setup

Clone the repo into a fresh ROS 2 workspace:

```bash
mkdir -p ~/so101_ws/src
cd ~/so101_ws/src
git clone https://github.com/thakkarnitin63/fireloop-isaac-ros2.git .
```

Expected structure after cloning:

```
so101_ws/
└── src/
    ├── so101_description       # URDF + meshes
    ├── so101_bringup           # Top-level launch
    ├── so101_moveit_config     # SRDF, kinematics, controllers
    ├── so101_perception        # Vision pipeline
    ├── so101_planning          # MotionPlanner (MoveItPy wrapper)
    └── so101_state_machine     # Behavior Tree node
```

---

## Build

From the workspace root:

```bash
cd ~/so101_ws

# 1. Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# 2. Build all packages
colcon build

# 3. Source the overlay
source install/setup.bash
```

> **Tip:** Add `source ~/so101_ws/install/setup.bash` to your `~/.bashrc` so you don't have to source it in every new terminal.

---

## Running the System

You will need **3 things running**: Isaac Sim, the bringup terminal, and the behavior tree terminal.

### Step 1 — Start Isaac Sim

1. Open NVIDIA Isaac Sim.
2. Load the provided **SO101 USD scene** (contains robot, red cup, green cup, and bin).
3. Press **Play** in the Isaac Sim toolbar to start the physics simulation. Joint states begin publishing on `/joint_states`.

### Step 2 — Launch Robot Bringup + MoveIt

In a new terminal:

```bash
source ~/so101_ws/install/setup.bash

ros2 launch so101_bringup bringup_moveit.launch.py \
    use_sim_time:=true \
    use_fake_hardware:=true
```

This starts:
- Robot state publisher (loads the SO101 URDF)
- `joint_trajectory_controller` and `gripper_controller` via `ros2_control`
- MoveIt 2 with **OMPL** and **Pilz Industrial Motion Planner** pipelines
- RViz for visualization

**Verify before continuing:**
- ✅ Robot model appears in RViz
- ✅ `ros2 control list_controllers` shows both controllers as `active`
- ✅ MoveIt motion planning works in the RViz MotionPlanning panel

### Step 3 — Run the Behavior Tree

In a **second terminal**:

```bash
source ~/so101_ws/install/setup.bash

ros2 launch so101_state_machine bt_pick_place.launch.py
```

The behavior tree will immediately start executing:

1. Open gripper
2. Call `/detect_cup` to find the red cup
3. Move to pre-grasp pose (OMPL)
4. Cartesian approach to grasp pose (Pilz LIN, falls back to OMPL)
5. Attach object in Isaac Sim + close gripper
6. Lift, translate, lower over the bin (TF-driven)
7. Detach + release

The full sequence takes roughly **30 to 45 seconds**.

---

## Visualization

Add these displays in RViz to monitor the system:

| Topic | Display Type | Purpose |
|-------|--------------|---------|
| `/perception/debug_image` | Image | Annotated detection overlay |
| `/perception/grasp_marker` | Marker | Arrow at the grasp pose |
| `/perception/pointcloud_base` | PointCloud2 | 3D cloud in `base_link` |
| `/detected_cup_pose` | Pose | Final grasp pose |

---

## ROS Interface

| Type | Name | Description |
|------|------|-------------|
| Service | `/detect_cup` | Triggers full perception pipeline (`std_srvs/Trigger`) |
| Topic | `/detected_cup_pose` | Grasp pose in `base_link` |
| Topic | `/pre_grasp_pose` | Approach pose in `base_link` |
| Topic | `/isaac_attach_cube` | `Bool` — attach/detach object in sim |
| Sub | `/camera/color/image_raw` | RGB image from Isaac camera |
| Sub | `/camera/depth/image_raw` | Depth image from Isaac camera |
| Sub | `/camera/camera_info` | Camera intrinsics |

---

## Troubleshooting

**No camera data received**
The perception node uses `BEST_EFFORT` QoS to match Isaac Sim's publishers. Check that the Isaac scene's camera prims are publishing on the expected topics:
```bash
ros2 topic hz /camera/color/image_raw
```

**Controllers not active**
Verify `use_sim_time:=true` is set everywhere, and that Isaac Sim is actually playing (not paused):
```bash
ros2 control list_controllers
```

**TF lookup failed in BT**
The transform tree needs a moment to stabilize after each motion. The BT catches these and returns `RUNNING`; if it persists, check that `robot_state_publisher` and Isaac Sim are both publishing TFs:
```bash
ros2 run tf2_tools view_frames
```

**Pilz Cartesian planning fails**
This is expected near kinematic limits. The BT automatically falls back to OMPL, so this is not fatal.

---

## Architecture (Quick View)

```
  Isaac Sim ──── /camera/* ────►  perception_node ──► /detect_cup
       │                              │                    │
       │                              ▼                    │
       │                         color_segmenter           │
       │                         shape_validator           │
       │                         depth_estimator           │
       │                         grasp_estimator           │
       │                              │                    │
       │                              ▼                    │
       │                         /detected_cup_pose        │
       │                              │                    │
       │                              ▼                    │
       │                          BT Node ────► MotionPlanner ──► MoveIt 2
       │                              │                              │
       └──────────── /joint_states ◄──┴──────────────────────────────┘
```

See `architecture_report.pdf` for the full design write-up.

---

## Author

**Nitin** — Built for the Fireloop ROS 2 + Isaac Sim assignment.
