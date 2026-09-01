import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class FakeAprilTagPublisher(Node):
    def __init__(self):
        super().__init__('fake_apriltag_publisher')

        self.publisher = self.create_publisher(
            Image,
            '/image_raw',
            10
        )

        self.bridge = CvBridge()

        # ==================================================
        # Image parameters
        # ==================================================

        self.width = 640
        self.height = 480

        # ==================================================
        # AprilTag parameters
        # ==================================================

        self.tag_id = 0

        # Physical tag size
        self.tag_size = 0.10

        # Actual tag size in the image
        self.tag_size_px = 200

        # Position of tag center
        self.tag_x = 420
        self.tag_y = 290

        # Rotation around camera optical axis
        self.tag_rotation = -90.0 + 20

        # ==================================================
        # Publish rate
        # ==================================================

        self.timer = self.create_timer(
            0.1,
            self.publish_image
        )

        self.get_logger().info(
            'Publishing fake AprilTag image on /camera/image_raw'
        )

    def generate_image(self):

        # ==================================================
        # Create white background
        # ==================================================

        image = np.ones(
            (self.height, self.width, 3),
            dtype=np.uint8
        ) * 255

        # ==================================================
        # AprilTag dictionary
        # ==================================================

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )

        # ==================================================
        # Generate AprilTag
        # ==================================================

        tag = np.zeros(
            (self.tag_size_px, self.tag_size_px),
            dtype=np.uint8
        )

        cv2.aruco.drawMarker(
            dictionary,
            self.tag_id,
            self.tag_size_px,
            tag,
            1
        )

        # ==================================================
        # Create larger canvas
        #
        # This prevents the corners from being cropped
        # when the tag is rotated.
        # ==================================================

        diagonal = int(
            np.ceil(
                np.sqrt(
                    self.tag_size_px ** 2 +
                    self.tag_size_px ** 2
                )
            )
        )

        canvas = np.ones(
            (diagonal, diagonal),
            dtype=np.uint8
        ) * 255

        # Center original tag on larger canvas

        offset_x = (
            diagonal - self.tag_size_px
        ) // 2

        offset_y = (
            diagonal - self.tag_size_px
        ) // 2

        canvas[
            offset_y:
            offset_y + self.tag_size_px,

            offset_x:
            offset_x + self.tag_size_px
        ] = tag

        # ==================================================
        # Rotate larger canvas
        # ==================================================

        center = (
            diagonal // 2,
            diagonal // 2
        )

        rotation_matrix = cv2.getRotationMatrix2D(
            center,
            self.tag_rotation,
            1.0
        )

        rotated = cv2.warpAffine(
            canvas,
            rotation_matrix,
            (diagonal, diagonal),
            flags=cv2.INTER_NEAREST,
            borderValue=255
        )

        # ==================================================
        # Crop rotated tag back to its bounding box
        # ==================================================

        # Find non-white pixels
        mask = rotated < 250

        if not np.any(mask):
            return image

        ys, xs = np.where(mask)

        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        rotated_tag = rotated[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]

        # ==================================================
        # Convert grayscale -> BGR
        # ==================================================

        tag_bgr = cv2.cvtColor(
            rotated_tag,
            cv2.COLOR_GRAY2BGR
        )

        tag_height, tag_width = tag_bgr.shape[:2]

        # ==================================================
        # Position rotated tag
        # ==================================================

        x = int(
            self.tag_x - tag_width / 2
        )

        y = int(
            self.tag_y - tag_height / 2
        )

        # ==================================================
        # Clip against image boundaries
        # ==================================================

        x1 = max(x, 0)
        y1 = max(y, 0)

        x2 = min(
            x + tag_width,
            self.width
        )

        y2 = min(
            y + tag_height,
            self.height
        )

        if x1 < x2 and y1 < y2:

            tag_x1 = x1 - x
            tag_y1 = y1 - y

            tag_x2 = tag_x1 + (x2 - x1)
            tag_y2 = tag_y1 + (y2 - y1)

            image[
                y1:y2,
                x1:x2
            ] = tag_bgr[
                tag_y1:tag_y2,
                tag_x1:tag_x2
            ]

        return image

    def publish_image(self):

        image = self.generate_image()

        msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.header.frame_id = 'camera_link'

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = FakeAprilTagPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()