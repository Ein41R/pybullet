import os
import time
from pathlib import Path

import cupy as cp
import pybullet as p
import pybullet_data
import pybullet_industrial as pi

#https://pybullet-industrial.readthedocs.io/en/latest/


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_path(env_name: str, default_path: Path) -> Path:
    raw_value = os.getenv(env_name)
    if not raw_value:
        return default_path

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


_load_env_file()
URDF_PATH = _resolve_path("PYBULLET_UR10E_URDF", PROJECT_ROOT / "Include" / "ur10e.urdf")

def get_movable_joint_indices(body_id):
    movable = []
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        joint_type = joint_info[2]
        if joint_type != p.JOINT_FIXED:
            movable.append(joint_index)
    return movable

def trace_trajectory(self):
    test_path = pi.build_box_path(
        center_position=[0, 0, 1.5],
        dimensions=[1,1],
        radius=0.3,
        orientation=p.getQuaternionFromEuler([cp.pi , 0, 0]),
        samples=1000
    )

    # while True:
    for pos, orn, _ in test_path:
        ur10e.goto_joint_positions(pos, orn)
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
pi.RobotBase.trace_trajectory = trace_trajectory


def goto_joint_positions(self, target_position, target_orientation, threshold=0.1, max_steps=240):
    """
    Move the robot to the target joint positions using position control.

    :param target_position: target end-effector position as a list or array.
    :param target_orientation: target end-effector orientation as a quaternion (x, y, z, w).
    :param threshold: Threshold for considering the joint position reached.
    :param max_steps: Maximum number of simulation steps to attempt.
    """
    step_count = 0
    reached = False
    self.set_endeffector_pose(target_position, target_orientation)

    while not reached:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

        cur_pos, cur_orn = self.get_endeffector_pose()
        pos_err = cp.linalg.norm(cp.array(target_position) - cp.array(cur_pos))
        orn_err = cp.linalg.norm(cp.array(target_orientation) - cp.array(cur_orn))
        step_count += 1
        if pos_err < threshold and orn_err < threshold or step_count > max_steps:
            reached = True
            print(f"Reached target position: {target_position}, orientation: {target_orientation} in {step_count} steps.\n")

pi.RobotBase.goto_joint_positions = goto_joint_positions

if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
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