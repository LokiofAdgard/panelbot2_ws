import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import Image
import numpy as np
import cv2

class ToFHeatmap(Node):
    def __init__(self):
        super().__init__('tof_heatmap')

        self.pub = self.create_publisher(Image, 'tof_heatmap', 10)
        self.sub = self.create_subscription(
            UInt16MultiArray,
            '/raw/tof',
            self.callback,
            10
        )

        self.get_logger().info("ToF Heatmap Visualizer Started")

    def callback(self, msg):
        data64 = msg.data[:64]
        # if len(msg.data) != 64:
        #     self.get_logger().warn(f"Expected 64 values, got {len(msg.data)}")
        #     return

        # Convert to numpy 8x8
        arr = np.array(data64, dtype=np.float32).reshape((8, 8))

        # Normalize to 0–255
        norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        # Apply Jet colormap
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Prepare ROS Image message
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "tof_link"
        img_msg.height = 8
        img_msg.width = 8
        img_msg.encoding = "rgb8"
        img_msg.step = 8 * 3
        img_msg.data = heatmap.flatten().tolist()

        self.pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ToFHeatmap()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
