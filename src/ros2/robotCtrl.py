import rclpy
from sensor_msgs.msg import JointState

class RobotController:
    def __init__(self):
        super().__init__("robotController")
        self.subscriber = self.create_subscription(
            JointState,
            "/joint_satates",
            self.joint_state_callback,
            10
        )

        self.publisher = self.create_publisher(
            JointState,
            "/follow_joint_trajectory/joint_ctrl",
            10,
        )

    def joint_state_callback(self, msg):
        pass

    def publish_joint_state(self, joint_state):
        self.publisher.publish(joint_state)