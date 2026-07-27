#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from scipy.spatial.transform import Rotation, Slerp


class ImuNoDrift(Node):

    def __init__(self):
        super().__init__("imu_no_drift")

        # Smooth offset quaternion
        self.offset = Rotation.identity()

        # Last raw quaternion
        self.last_raw = None

        # Smoothing factor (0.0 = no smoothing, 1.0 = instant)
        self.alpha = 0.05

        self.sub = self.create_subscription(
            Float32MultiArray,
            "raw/imu",
            self.imu_callback,
            20
        )

        self.br = TransformBroadcaster(self)
        self.get_logger().info("IMU smooth relative TF broadcaster started")


    def imu_callback(self, msg):

        if len(msg.data) != 11:
            return

        # Raw quaternion
        qw, qx, qy, qz = msg.data[0:4]
        q_raw = Rotation.from_quat([qx, qy, qz, qw])

        calib = msg.data[10]

        # First frame
        if self.last_raw is None:
            self.last_raw = q_raw
            return

        # Detect magnetometer snap (calib == 2)
        if calib == 2:
            # Compute delta between last and current
            delta = q_raw * self.last_raw.inv()

            # SLERP from current offset → delta
            key_times = [0, 1]
            key_rots = Rotation.concatenate([self.offset, delta])
            slerp = Slerp(key_times, key_rots)

            # Blend by alpha
            self.offset = slerp(self.alpha)

        # Apply smooth correction
        q_corrected = q_raw * self.offset.inv()

        # Update last raw
        self.last_raw = q_raw

        self.publish_tf(q_corrected)


    def publish_tf(self, orientation):

        q = orientation.as_quat()

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()

        t.header.frame_id = "odom"
        t.child_frame_id  = "base_link"

        # Fixed translation (no drift)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Smooth corrected IMU orientation
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ImuNoDrift()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
