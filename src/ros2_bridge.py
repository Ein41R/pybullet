import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class PyBulletRosBridge(Node):
    ### initializing Ros 2 node to bridge the pybullet simulation with ros 2 topics
    def __init__(self, robot_id, joint_indices, joint_names): #constructor
        super().__init__("pybullet_bridge")
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.joint_names = joint_names
        self.target_positions = [0.0] * len(joint_indices)

        self.create_subscription(#subscribe to the joint trajectory topic (created by controller)
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            self.on_trajectory,
            10,
        )
        self.joint_states = self.create_publisher(JointState, "/joint_states", 10) #punlish own joint states

    ### messages are tuples of points aka. joint_names to positions, velocities, accelerations, effort
    ### 
    def on_trajectory(self, message):
        if not message.points: #ends if message is empty
            return

        point = message.points[-1] #selects last point
        positions = dict(zip(message.joint_names, point.positions)) #are we sure we can do this? dict removes duplicates    
        self.target_positions = [
            positions.get(name, target)
            for name, target in zip(self.joint_names, self.target_positions)
        ]

    def apply_targets(self, pybullet):
        for joint_index, target in zip(self.joint_indices, self.target_positions):
            pybullet.setJointMotorControl2(
                self.robot_id,
                joint_index,
                pybullet.POSITION_CONTROL,
                targetPosition=target,
                force=150.0,
            )

    def publish_joint_states(self, pybullet):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = self.joint_names
        message.position = [
            pybullet.getJointState(self.robot_id, joint_index)[0]
            for joint_index in self.joint_indices
        ]
        self.joint_states.publish(message)