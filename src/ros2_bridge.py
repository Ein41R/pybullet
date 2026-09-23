from rclpy.node import Node
from sensor_msgs.msg import JointState #https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/JointState.html
import pybullet as p

# TLDR:
# publishes     »   /joint_states
# subscribes    »   /follow_joint_trajectory/joint_ctrl

class PyBulletRosBridge(Node):
    def __init__(self, joint_names, joint_indices, ur10e):
        self.joint_names = joint_names
        self.joint_indices = joint_indices
        self.ur10e = ur10e

        super().__init__("pybullet_bridge")
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / 60.0, self.broadcast)

        self.subscriptions = self.create_subscription(
            JointState,
            "/follow_joint_trajectory/joint_ctrl",
            self.callback,
            10)
        
    
    def broadcast(self):
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

    def callback(self, msg):
        for _, name in enumerate(msg.name):
            if name in self.joint_names:
                joint_index = self.joint_names.index(name)
                p.setJointMotorControl2(
                    self.ur10e,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=msg.position[i]
                )
