import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
import numpy as np


class InitialPosPublisher(Node):
    def __init__(self):
        super().__init__('initial_pos_publisher')

        # Parameters (tunable from launch/YAML)
        self.declare_parameter("initial_x", 1.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_yaw", 1.5708)

        x = self.get_parameter("initial_x").value
        y = self.get_parameter("initial_y").value
        yaw = self.get_parameter("initial_yaw").value

        # Static TF broadcaster
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)

        # Publish once
        self.publish_tf(x, y, yaw)
        self.get_logger().info(
            f"Published initial map→odom transform: "
            f"x={x:.3f}, y={y:.3f}, yaw={np.degrees(yaw):.1f}°"
        )

    def publish_tf(self, x, y, yaw):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        t.transform.rotation.z = np.sin(yaw / 2.0)
        t.transform.rotation.w = np.cos(yaw / 2.0)

        self.tf_static.sendTransform(t)


def main(args=None):
        rclpy.init(args=args)
        node = InitialPosPublisher()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
        main()
