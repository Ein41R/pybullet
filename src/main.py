import os
import time
from pathlib import Path

import cupy as cp
import pybullet as p
import pybullet_data
import pybullet_industrial as pi

from dotenv import load_dotenv

# Loading environment variables from .env.development file in project root
home = Path.home()
load_dotenv(f"{home}/Dev/pybullet/.env.development")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYBULLET_DATA_PATH = os.getenv("PYBULLET_DATA_PATH") or pybullet_data.getDataPath()
URDF_PATH = os.getenv("URDF_PATH") or str(PROJECT_ROOT / "assets" / "ur10e.urdf")
ROS_URDF_PACKAGE_PATH = PROJECT_ROOT / "assets" / "Universal_Robots" / "install" / "ur_description" / "share" / "ur_description"

urdf_file = Path(URDF_PATH)
if not urdf_file.exists():
    raise FileNotFoundError(f"URDF file not found: {urdf_file}")

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
    p.connect(p.GUI)

    p.setAdditionalSearchPath(PYBULLET_DATA_PATH)

    # Set physics parameters
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    p.loadURDF("plane.urdf", basePosition=[0, 0, -0.001])
    p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.001], baseOrientation=p.getQuaternionFromEuler([0, 0, cp.pi / 2]))
    sphere = p.loadURDF("sphere_small.urdf", basePosition=[0, 1, 1.5], globalScaling=1.5)

    #the sphere is too fast
    p.changeDynamics(sphere, -1, linearDamping=0.99, angularDamping=0.99)

    start_position = cp.array([0, 0, 0.585])
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    ur10e = pi.RobotBase(str(URDF_PATH), start_position, start_orientation)

    # Tuple[List[str], List[int]], being joint names and joint indices
    movable_joints = ur10e.get_moveable_joints()
    
    while True:
        sphere_pos, sphere_orn = p.getBasePositionAndOrientation(sphere)
        ur10e.goto_joint_positions(sphere_pos, sphere_orn, max_steps=10)