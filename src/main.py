import os
import time
from pathlib import Path

import cupy as cp
import pybullet as p
import pybullet_data
import pybullet_industrial as pi


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