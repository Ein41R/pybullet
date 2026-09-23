import os
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import rclpy

from dotenv import load_dotenv

from ros2_bridge import PyBulletRosBridge
from sensor_msgs.msg import JointState

# Loading environment variables from .env.development file in project root
home = Path.home()
load_dotenv(f"{home}/Dev/pybullet/.env.development")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYBULLET_DATA_PATH = os.getenv("PYBULLET_DATA_PATH") or pybullet_data.getDataPath()
URDF_PATH = os.getenv("URDF_PATH") or str(PROJECT_ROOT / "assets" / "ur10e.urdf")
ROS_URDF_PACKAGE_PATH = PROJECT_ROOT / "assets" / "Universal_Robots" / "install" / "ur_description" / "share" / "ur_description"

# check if file exists, idk copilot added it
urdf_file = Path(URDF_PATH)
if not urdf_file.exists():
    raise FileNotFoundError(f"URDF file not found: {urdf_file}")

# replacing package:// ros style URI with absolute path, since I have no fucking idea what is causing the issue
if "package://ur_description/" in urdf_file.read_text(encoding="utf-8"):
    fixed_urdf = PROJECT_ROOT / "assets" / "ur10e_fixed.urdf"
    fixed_urdf.write_text(
        urdf_file.read_text(encoding="utf-8").replace(
            "package://ur_description/",
            str(ROS_URDF_PACKAGE_PATH.as_posix()) + "/",
        ),
        encoding="utf-8",
    )
    URDF_PATH = str(fixed_urdf)


if __name__ == "__main__":
    #setup pybullet simulation
    p.connect(p.GUI)

    # Set the additional search path for PyBullet data
    p.setAdditionalSearchPath(PYBULLET_DATA_PATH)

    # Set physics parameters
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # from old deprecaed project copy pasted
    p.loadURDF("plane.urdf", basePosition=[0, 0, -0.001])
    p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.001], baseOrientation=p.getQuaternionFromEuler([0, 0, 1.57079632679]))
    sphere = p.loadURDF("sphere_small.urdf", basePosition=[0, 1, 1.5], globalScaling=1.5)

    #the sphere is too fast
    p.changeDynamics(sphere, -1, linearDamping=0.99, angularDamping=0.99)

    start_position = [0, 0, 0.585]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    ur10e = p.loadURDF(
        str(URDF_PATH),
        basePosition=start_position,
        baseOrientation=start_orientation,
        useFixedBase=True,
    )

    # hardcoding ur10e joint specification.
    joint_names = [
        "shoulder_pan_joint", # base Joint
        "shoulder_lift_joint", # Shoulder Joint
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    joint_indices = [
        p.getJointInfo(ur10e, i)[0] for i in range(p.getNumJoints(ur10e))
    ]

    # Initialize the ROS 2 node.
    rclpy.init()
    bridge = PyBulletRosBridge(joint_names, joint_indices, ur10e)

    try:
        while rclpy.ok() and p.isConnected():
            rclpy.spin_once(bridge, timeout_sec=0.0)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        p.disconnect()