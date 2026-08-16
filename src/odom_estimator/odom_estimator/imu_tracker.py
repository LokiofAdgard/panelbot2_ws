import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from mros_interfaces.msg import Imu
import numpy as np
import tf2_ros


class IMUTracker(Node):
    def __init__(self):
        super().__init__("imu_tracker")

        # State
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0
        self.last_time = None

        # Bias parameters (optional)
        self.declare_parameter("gyro_bias_z", 0.0)
        self.declare_parameter("accel_bias_x", 0.0)
        self.declare_parameter("accel_bias_y", 0.0)

        # ROS interfaces
        self.sub = self.create_subscription(Imu, "/imu/data", self.cb, 50)
        self.pub_odom = self.create_publisher(Odometry, "/imu/odom", 10)
        self.tf = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info("IMU Tracker Node Started")

    # --------------------------------------------------------
    # CALLBACK
    # --------------------------------------------------------
    def cb(self, msg: Imu):

        # Always use ROS time (safe, no header needed)
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_time is None:
            self.last_time = now
            return

        dt = now - self.last_time
        self.last_time = now

        if dt <= 0:
            return

        # Read IMU data
        ax = msg.accel.x - self.get_parameter("accel_bias_x").value
        ay = msg.accel.y - self.get_parameter("accel_bias_y").value
        gz = msg.gyro.z - self.get_parameter("gyro_bias_z").value

        # Update yaw
        self.theta += gz * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # Rotate accel into world frame
        ax_world = ax * np.cos(self.theta) - ay * np.sin(self.theta)
        ay_world = ax * np.sin(self.theta) + ay * np.cos(self.theta)

        # Integrate velocity
        self.vx += ax_world * dt
        self.vy += ay_world * dt

        # Integrate position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Debug
        self.get_logger().info(
            f"[IMU] dt={dt:.3f} θ={np.degrees(self.theta):.1f}° "
            f"ax={ax:.2f} ay={ay:.2f} vx={self.vx:.2f} vy={self.vy:.2f} "
            f"x={self.x:.2f} y={self.y:.2f}"
        )

        # Publish using ROS time
        stamp = self.get_clock().now().to_msg()
        self.publish_odom(stamp)

    # --------------------------------------------------------
    # PUBLISH ODOM + TF
    # --------------------------------------------------------
    def publish_odom(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint_imu"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        odom.pose.pose.orientation.z = np.sin(self.theta / 2)
        odom.pose.pose.orientation.w = np.cos(self.theta / 2)

        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.angular.z = self.theta

        self.pub_odom.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint_imu"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = np.sin(self.theta / 2)
        t.transform.rotation.w = np.cos(self.theta / 2)

        self.tf.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = IMUTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
