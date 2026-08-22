import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException


class AprilTagDriftChecker(Node):

    def __init__(self):
        super().__init__('apriltag_drift_checker')

        # Camera feed
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.map_frame = 'map'
        self.base_frame = 'base_footprint_ecc'

        # Parameters
        self.tag_id = 0
        self.tag_size = 0.10

        self.declare_parameter("x_axis", "-camera_x")
        self.declare_parameter("y_axis", "-camera_y")
        self.declare_parameter("calibration_scale", 0.86)

        self.x_axis = self.get_parameter("x_axis").value
        self.y_axis = self.get_parameter("y_axis").value
        self.calibration_scale = float(self.get_parameter("calibration_scale").value)

        # Camera intrinsics
        self.camera_matrix = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # AprilTag detector
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.latest_tag_pose = None

        # 1 Hz drift check
        self.timer = self.create_timer(1.0, self.validate)


    # ------------------------------------------------------
    # Axis conversion
    # ------------------------------------------------------
    def get_axis_value(self, value_x, value_y, axis):
        if axis == "camera_x": return value_x
        if axis == "-camera_x": return -value_x
        if axis == "camera_y": return value_y
        if axis == "-camera_y": return -value_y
        raise ValueError(f"Invalid axis: {axis}")


    # ------------------------------------------------------
    # Quaternion → yaw
    # ------------------------------------------------------
    def quaternion_to_yaw(self, q):
        return np.arctan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )


    # ------------------------------------------------------
    # Normalize angle
    # ------------------------------------------------------
    def normalize(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))


    # ------------------------------------------------------
    # AprilTag detection
    # ------------------------------------------------------
    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if ids is None:
            return

        # find tag
        tag_index = None
        for i, detected_id in enumerate(ids.flatten()):
            if detected_id == self.tag_id:
                tag_index = i
                break
        if tag_index is None:
            return

        # solvePnP
        half = self.tag_size / 2.0
        object_points = np.array([
            [-half, -half, 0.0],
            [ half, -half, 0.0],
            [ half,  half, 0.0],
            [-half,  half, 0.0]
        ], dtype=np.float64)

        image_points = corners[tag_index].reshape(4, 2).astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return

        # camera → tag translation
        tag_x = float(tvec[0][0]) * self.calibration_scale
        tag_y = float(tvec[1][0]) * self.calibration_scale

        # rotation
        R, _ = cv2.Rodrigues(rvec)
        yaw = np.arctan2(R[1,0], R[0,0])

        # axis conversion
        world_x = self.get_axis_value(tag_x, tag_y, self.x_axis)
        world_y = self.get_axis_value(tag_x, tag_y, self.y_axis)

        # ------------------------------------------------------
        # CORRECTED: DO NOT INVERT TRANSLATION
        #
        # AprilTag pose in MAP frame:
        #
        # map_x = world_x
        # map_y = world_y
        #
        # Only yaw is inverted (camera → tag → map)
        # ------------------------------------------------------
        map_x = world_x
        map_y = world_y
        map_yaw = yaw

        self.latest_tag_pose = (map_x, map_y, map_yaw)


    # ------------------------------------------------------
    # Drift validation
    # ------------------------------------------------------
    def validate(self):

        if self.latest_tag_pose is None:
            self.get_logger().info("Waiting for AprilTag...")
            return

        # TF: map → base_footprint_ecc
        try:
            tf_map_base = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
        except TransformException:
            self.get_logger().warn("Cannot get map → base_footprint_ecc")
            return

        ecc_x = tf_map_base.transform.translation.x
        ecc_y = tf_map_base.transform.translation.y
        ecc_yaw = self.quaternion_to_yaw(tf_map_base.transform.rotation)

        tag_x, tag_y, tag_yaw = self.latest_tag_pose
        self.publish_april_tf(tag_x, tag_y, tag_yaw)

        dx = tag_x - ecc_x
        dy = tag_y - ecc_y
        dyaw = self.normalize(tag_yaw - ecc_yaw) - np.pi

        self.get_logger().info("--------------------------------------------------")
        self.get_logger().info(f"TAG (map): x={tag_x:+.3f}  y={tag_y:+.3f}  yaw={np.degrees(tag_yaw):+.2f}")
        self.get_logger().info(f"ECC (map): x={ecc_x:+.3f}  y={ecc_y:+.3f}  yaw={np.degrees(ecc_yaw):+.2f}")
        self.get_logger().info(f"DRIFT: dx={dx:+.3f}  dy={dy:+.3f}  dist={np.sqrt(dx*dx+dy*dy):.3f}  yaw={np.degrees(dyaw):+.2f}")
        self.get_logger().info("--------------------------------------------------")

    def publish_april_tf(self, x, y, yaw):
        t = tf2_ros.TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'april_live'

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = float(np.sin(yaw / 2.0))
        t.transform.rotation.w = float(np.cos(yaw / 2.0))

        self.tf_broadcaster.sendTransform(t)



def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDriftChecker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
