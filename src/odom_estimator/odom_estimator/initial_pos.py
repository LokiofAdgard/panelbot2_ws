import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge
import tf2_ros


class InitialPosPublisher(Node):

    def __init__(self):
        super().__init__('initial_pos_publisher')

        # ==================================================
        # ROS
        # ==================================================

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()

        # Static TF broadcaster
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)

        # Only initialize once
        self.pose_initialized = False

        # ==================================================
        # Axis configuration
        # ==================================================

        # Camera/image coordinate convention:
        #
        # Camera X = right
        # Camera Y = down
        #
        # Set these to define how camera coordinates map
        # to your desired world/map coordinates.
        #
        # Examples:
        #
        # x_axis = "camera_x"
        # y_axis = "camera_y"
        #
        # x_axis = "-camera_x"
        # y_axis = "camera_y"
        #
        # x_axis = "camera_y"
        # y_axis = "-camera_x"
        #
        # etc.

        self.declare_parameter(
            "x_axis",
            "-camera_x"
        )

        self.declare_parameter(
            "y_axis",
            "-camera_y"
        )

        self.x_axis = self.get_parameter(
            "x_axis"
        ).value

        self.y_axis = self.get_parameter(
            "y_axis"
        ).value

        valid_axes = [
            "camera_x",
            "-camera_x",
            "camera_y",
            "-camera_y"
        ]

        if self.x_axis not in valid_axes:
            raise ValueError(
                f"Invalid x_axis: {self.x_axis}"
            )

        if self.y_axis not in valid_axes:
            raise ValueError(
                f"Invalid y_axis: {self.y_axis}"
            )

        # ==================================================
        # Distance calibration
        # ==================================================

        # Multiplier applied to the estimated position.
        #
        # 1.0 = no correction
        #
        # Example:
        #
        # detector says 0.25 m
        # actual distance is 0.50 m
        #
        # calibration_scale = 2.0
        #
        # estimated:
        #
        # 0.25 * 2.0 = 0.50 m

        self.declare_parameter(
            "calibration_scale",
            0.86
        )

        self.calibration_scale = float(
            self.get_parameter(
                "calibration_scale"
            ).value
        )

        # ==================================================
        # Camera parameters
        # ==================================================

        self.image_width = 640
        self.image_height = 480

        self.fx = 500.0
        self.fy = 500.0

        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0

        self.camera_matrix = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros(
            (5, 1),
            dtype=np.float64
        )

        # ==================================================
        # AprilTag parameters
        # ==================================================

        self.tag_id = 0

        # Physical size of AprilTag in metres

        self.tag_size = 0.10

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )

        self.parameters = (
            cv2.aruco.DetectorParameters_create()
        )

        self.get_logger().info(
            'Waiting for AprilTag to determine initial pose...'
        )

        self.get_logger().info(
            f'Axis configuration: '
            f'x={self.x_axis}, '
            f'y={self.y_axis}'
        )

        self.get_logger().info(
            f'Calibration scale: '
            f'{self.calibration_scale:.4f}'
        )

    # ======================================================
    # Convert camera coordinate to configured world axis
    # ======================================================

    def get_axis_value(self, value_x, value_y, axis):

        if axis == "camera_x":
            return value_x

        if axis == "-camera_x":
            return -value_x

        if axis == "camera_y":
            return value_y

        if axis == "-camera_y":
            return -value_y

        raise ValueError(
            f"Invalid axis: {axis}"
        )

    # ======================================================
    # Image callback
    # ======================================================

    def image_callback(self, msg):

        if self.pose_initialized:
            return

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================================
        # Detect AprilTag
        # ==================================================

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.parameters
        )

        if ids is None:
            return

        # ==================================================
        # Find requested tag
        # ==================================================

        tag_index = None

        for i, detected_id in enumerate(
            ids.flatten()
        ):

            if detected_id == self.tag_id:
                tag_index = i
                break

        if tag_index is None:
            return

        self.get_logger().info(
            f'AprilTag {self.tag_id} detected. '
            f'Estimating initial pose...'
        )

        # ==================================================
        # Image points
        # ==================================================

        image_points = corners[tag_index].reshape(
            4,
            2
        ).astype(np.float64)

        # ==================================================
        # Tag coordinates
        # ==================================================

        half = self.tag_size / 2.0

        object_points = np.array([
            [-half, -half, 0.0],
            [ half, -half, 0.0],
            [ half,  half, 0.0],
            [-half,  half, 0.0]
        ], dtype=np.float64)

        # ==================================================
        # Pose estimation
        # ==================================================

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            self.get_logger().warn(
                'AprilTag pose estimation failed'
            )
            return

        # ==================================================
        # Camera -> Tag position
        # ==================================================

        tag_x = float(tvec[0][0])
        tag_y = float(tvec[1][0])
        tag_z = float(tvec[2][0])

        # ==================================================
        # Apply distance calibration
        # ==================================================

        tag_x *= self.calibration_scale
        tag_y *= self.calibration_scale
        tag_z *= self.calibration_scale

        # ==================================================
        # Rotation
        # ==================================================

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 +
            rotation_matrix[1, 0] ** 2
        )

        if sy > 1e-6:

            roll = np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )

            pitch = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            yaw = np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )

        else:

            roll = np.arctan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1]
            )

            pitch = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            yaw = 0.0

        # ==================================================
        # Convert to configured axes
        # ==================================================

        world_tag_x = self.get_axis_value(
            tag_x,
            tag_y,
            self.x_axis
        )

        world_tag_y = self.get_axis_value(
            tag_x,
            tag_y,
            self.y_axis
        )

        # ==================================================
        # Calculate initial map -> odom
        # ==================================================

        initial_x = -world_tag_y
        initial_y = -world_tag_x

        initial_yaw = yaw - np.pi/2

        # ==================================================
        # Log
        # ==================================================

        self.get_logger().info(
            f'AprilTag pose: '
            f'camera_x={tag_x:.3f}, '
            f'camera_y={tag_y:.3f}, '
            f'camera_z={tag_z:.3f} m'
        )

        self.get_logger().info(
            f'Configured position: '
            f'x={world_tag_x:.3f}, '
            f'y={world_tag_y:.3f} m'
        )

        self.get_logger().info(
            f'Rotation: '
            f'roll={np.degrees(roll):.1f}°, '
            f'pitch={np.degrees(pitch):.1f}°, '
            f'yaw={np.degrees(yaw):.1f}°'
        )

        self.get_logger().info(
            f'INITIAL POSE: '
            f'x={initial_x:.3f}, '
            f'y={initial_y:.3f}, '
            f'yaw={np.degrees(initial_yaw):.1f}°'
        )

        # ==================================================
        # Publish static TF
        # ==================================================

        self.publish_tf(
            initial_x,
            initial_y,
            initial_yaw
        )

        self.pose_initialized = True

    # ======================================================
    # Static TF
    # ======================================================

    def publish_tf(
        self,
        x,
        y,
        yaw
    ):

        t = TransformStamped()

        t.header.stamp = (
            self.get_clock().now().to_msg()
        )

        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0

        t.transform.rotation.z = float(
            np.sin(yaw / 2.0)
        )

        t.transform.rotation.w = float(
            np.cos(yaw / 2.0)
        )

        self.tf_static.sendTransform(t)

    # ======================================================


def main(args=None):

    rclpy.init(args=args)

    node = InitialPosPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()