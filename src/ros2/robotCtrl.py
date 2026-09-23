import rclpy


if name == "__main__":
    rclpy.init()
    node = rclpy.create_node("pybullet_bridge")
    node.get_logger().info("PyBullet ROS 2 bridge node started.")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()