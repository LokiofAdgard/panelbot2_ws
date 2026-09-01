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

        # Camera topic to listen to
        CAM_TOPIC = "/image_raw"

        # Expected AprilTag ID
        TAG_ID = 0

        # Physical tag size (meters)
        TAG_SIZE = 0.048

        # Camera intrinsics
        IMG_W = 640
        IMG_H = 480
        FX = 500.0
        FY = 500.0
        CX = IMG_W / 2.0
        CY = IMG_H / 2.0

        # Axis mapping: camera_x, -camera_x, camera_y, -camera_y
        X_AXIS = "camera_y"
        Y_AXIS = "camera_x"
        INVERT_YAW = True

        # Calibration multiplier (distance correction)
        CALIB_SCALE = 1.0

        # Manual offsets applied AFTER estimation
        OFFSET_X = 0.0
        OFFSET_Y = 0.0
        OFFSET_YAW = 90.0  # degrees

        # ==================================================
        # Store settings
        # ==================================================

        self.tag_id = TAG_ID
        self.tag_size = TAG_SIZE
        self.x_axis = X_AXIS
        self.y_axis = Y_AXIS
        self.invert_yaw = INVERT_YAW
        self.calibration_scale = CALIB_SCALE
        self.offset_x = OFFSET_X
        self.offset_y = OFFSET_Y
        self.offset_yaw = np.radians(OFFSET_YAW)

        # Camera matrix
        self.camera_matrix = np.array([
            [FX, 0.0, CX],
            [0.0, FY, CY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # AprilTag dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        self.parameters = cv2.aruco.DetectorParameters_create()

        # ROS interfaces
        self.bridge = CvBridge()
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self.pose_initialized = False

        self.subscription = self.create_subscription(
            Image,
            CAM_TOPIC,
            self.image_callback,
            10
        )

        self.get_logger().info("InitialPosPublisher ready. Waiting for AprilTag...")

    # ======================================================
    # Axis conversion helper
    # ======================================================

    def get_axis_value(self, cam_x, cam_y, axis):
        if axis == "camera_x":
            return cam_x
        if axis == "-camera_x":
            return -cam_x
        if axis == "camera_y":
            return cam_y
        if axis == "-camera_y":
            return -cam_y
        raise ValueError(f"Invalid axis: {axis}")

    # ======================================================
    # Image callback
    # ======================================================

    def image_callback(self, msg):

        if self.pose_initialized:
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

        if ids is None:
            return

        # Find tag
        ids = ids.flatten()
        if self.tag_id not in ids:
            return

        idx = np.where(ids == self.tag_id)[0][0]
        image_points = corners[idx].reshape(4, 2).astype(np.float64)

        # Tag corner coordinates
        half = self.tag_size / 2.0
        object_points = np.array([
            [-half, -half, 0.0],
            [ half, -half, 0.0],
            [ half,  half, 0.0],
            [-half,  half, 0.0]
        ], dtype=np.float64)

        # SolvePnP
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            self.get_logger().warn("Pose estimation failed")
            return


        t_cam_tag = tvec.reshape(3, 1) * self.calibration_scale

        R_cam_tag, _ = cv2.Rodrigues(rvec)

        R_tag_cam = R_cam_tag.T
        t_tag_cam = -R_tag_cam @ t_cam_tag

        tag_x = float(t_tag_cam[0, 0])
        tag_y = float(t_tag_cam[1, 0])
        tag_z = float(t_tag_cam[2, 0])

        # Yaw of CAMERA relative to TAG
        yaw = np.arctan2(R_tag_cam[1, 0], R_tag_cam[0, 0])

        # Convert AprilTag-frame axes into map axes
        world_x = self.get_axis_value(tag_x, tag_y, self.x_axis)
        world_y = self.get_axis_value(tag_x, tag_y, self.y_axis)

        # Apply configured offsets
        init_x = world_x + self.offset_x
        init_y = world_y + self.offset_y
        init_yaw = yaw + self.offset_yaw

        if self.invert_yaw:
            init_yaw = -init_yaw

        # Optional: keep yaw normalized
        init_yaw = np.arctan2(np.sin(init_yaw), np.cos(init_yaw))

        # Publish TF
        self.publish_tf(init_x, init_y, init_yaw)

        self.get_logger().info(
            f"Initial pose set: x={init_x:.3f}, y={init_y:.3f}, yaw={np.degrees(init_yaw):.1f}°"
        )

        self.pose_initialized = True

    # ======================================================
    # Static TF publisher
    # ======================================================

    def publish_tf(self, x, y, yaw):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = np.sin(yaw / 2.0)
        t.transform.rotation.w = np.cos(yaw / 2.0)

        self.tf_static.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = InitialPosPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
