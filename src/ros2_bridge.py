from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import JointState #https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/JointState.html
from control_msgs.action import FollowJointTrajectory #https://docs.ros.org/en/noetic/api/control_msgs/html/action/FollowJointTrajectory.html
import pybullet as p

# TLDR:
# publishes     »   /joint_states
# subscribes    »   /follow_joint_trajectory/joint_ctrl

# Legend:
# DJC = Direct Joint Control
# TJC = Trajectory Joint Control

class PyBulletRosBridge(Node):
    def __init__(self, joint_names, joint_indices, ur10e):
        self.joint_names = joint_names
        self.joint_indices = joint_indices
        self.ur10e = ur10e

        super().__init__("pybullet_bridge")

        #topic publisher for joint states
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / 60.0, self.publish_joint_states)

        #topic subscriber for direct joint control (DJC)
        self.subscription = self.create_subscription(
            JointState,
            "/follow_joint_trajectory/joint_ctrl",
            self.subscribe_DJC,
            10)

        #action server for trajectory control (TJC)
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/follow_joint_trajectory",
            self.respond_tjc
        )
    
    def publish_joint_states(self):
        joint_states = [
            p.getJointState(self.ur10e, joint_index)
            for joint_index in self.joint_indices
        ]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [state[0] for state in joint_states]
        msg.velocity = [state[1] for state in joint_states]
        msg.effort = [state[3] for state in joint_states]
        self.publisher.publish(msg)

    def subscribe_DJC(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                joint_index = self.joint_names.index(name)
                p.setJointMotorControl2(
                    self.ur10e,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=msg.position[i]
                )

    def respond_tjc(self, goal_handle):
        # Implement trajectory control logic here
        # For now, we will just accept the goal and return success
        goal_handle.succeed()
        return FollowJointTrajectory.Result()  # Return an empty JointState message for now