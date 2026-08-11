import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf2_ros
import threading

# ============================================================
# SE(2) ESTIMATOR (ECC-based)
# ============================================================
class SE2Estimator:
    def __init__(self):
        self.warp_mode = cv2.MOTION_EUCLIDEAN
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            1e-5
        )

    def estimate(self, img1, img2):
        h, w = img1.shape
        cx, cy = w / 2.0, h / 2.0

        warp = np.eye(2, 3, dtype=np.float32)

        # Gaussian falloff mask
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        dist2 = xv**2 + yv**2
        mask = np.exp(-dist2 / 0.25)
        mask = (mask * 255).astype(np.uint8)

        try:
            cc, warp = cv2.findTransformECC(
                img1, img2, warp,
                self.warp_mode,
                self.criteria,
                mask,
                5
            )
        except cv2.error:
            return 0.0, 0.0, 0.0, 0.0

        dx = warp[0, 2]
        dy = warp[1, 2]
        theta = np.arctan2(warp[1, 0], warp[0, 0])
        if abs(theta) < 0.0004:
            theta = 0.0

        # rotation compensation
        dx_corr = dx - (cx * (1 - np.cos(theta)) + cy * np.sin(theta))
        dy_corr = dy - (cy * (1 - np.cos(theta)) - cx * np.sin(theta))

        return dx_corr, dy_corr, theta, cc


# ============================================================
# ODOMETRY INTEGRATOR (relative motion)
# ============================================================
class OdometryIntegrator:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def update(self, dy, dx, dtheta):
        global_dx = dx * np.cos(self.theta) - dy * np.sin(self.theta)
        global_dy = dx * np.sin(self.theta) + dy * np.cos(self.theta)

        self.x += global_dx
        self.y += global_dy

        self.theta += dtheta
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_pose(self):
        return self.x, self.y, self.theta


# ============================================================
# CALIBRATION
# ============================================================
class PixelToMeter:
    def __init__(self, px_per_meter=2175):
        self.scale = px_per_meter

    def to_m(self, px):
        return px / self.scale


# ============================================================
# IMAGE PREPROCESSOR
# ============================================================
class ImagePreprocessor:
    def __init__(self):
        self.crop_w = 470
        self.crop_h = 480

    def process(self, img_gray):
        h, w = img_gray.shape
        cx, cy = w // 2, h // 2

        x1 = cx - self.crop_w // 2
        y1 = cy - self.crop_h // 2
        x2 = cx + self.crop_w // 2
        y2 = cy + self.crop_h // 2

        img = img_gray[y1:y2, x1:x2]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img)

        sobelx = cv2.Sobel(img_eq, cv2.CV_32F, 1, 0, ksize=7)
        sobely = cv2.Sobel(img_eq, cv2.CV_32F, 0, 1, ksize=7)
        grad = cv2.magnitude(sobelx, sobely)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return grad


# ============================================================
# COMBINED NODE: ECC + INITIAL POSE TRACKING
# ============================================================
class SE2OdomTrackerNode(Node):
    def __init__(self):
        super().__init__('se2_odom_tracker_node')

        # Camera subscription
        self.sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )

        # Initial pose subscription
        self.create_subscription(
            Odometry,
            'initial_odom',
            self.initial_callback,
            10
        )

        # Output
        self.odom_pub = self.create_publisher(Odometry, '/tracked_odom', 10)
        self.proc_img_pub = self.create_publisher(Image, '/processed_image2', 10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.bridge = CvBridge()

        # ECC pipeline
        self.prev = None
        self.estimator = SE2Estimator()
        self.odom = OdometryIntegrator()
        self.calib = PixelToMeter()
        self.preproc = ImagePreprocessor()

        # Initial pose
        self.initial_received = False
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_theta = 0.0

        # Threading
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = None

        self.worker = threading.Thread(target=self.process_loop, daemon=True)
        self.worker.start()

    # ---------------------------------------------------------
    # INITIAL POSE CALLBACK
    # ---------------------------------------------------------
    def initial_callback(self, msg):
        if self.initial_received:
            return

        self.init_x = msg.pose.pose.position.x
        self.init_y = msg.pose.pose.position.y

        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.init_theta = 2 * np.arctan2(qz, qw)

        self.initial_received = True
        self.get_logger().info(
            f"Initial pose set: x={self.init_x:.3f}, y={self.init_y:.3f}, θ={np.degrees(self.init_theta):.2f}"
        )

    # ---------------------------------------------------------
    # FAST CAMERA CALLBACK
    # ---------------------------------------------------------
    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        with self.lock:
            self.latest_frame = img
            self.latest_stamp = msg.header.stamp

    # ---------------------------------------------------------
    # WORKER LOOP
    # ---------------------------------------------------------
    def process_loop(self):
        while rclpy.ok():
            frame = None
            stamp = None

            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    stamp = self.latest_stamp
                    self.latest_frame = None

            if frame is None or not self.initial_received:
                continue

            frame = self.preproc.process(frame)
            img = cv2.resize(frame, None, fx=0.5, fy=0.5)

            if self.prev is None:
                self.prev = img
                continue

            dx, dy, dtheta, score = self.estimator.estimate(self.prev, img)

            if score < 0.5:
                continue

            dx *= 2.0
            dy *= 2.0

            self.odom.update(dx, dy, dtheta)
            rel_x_px, rel_y_px, rel_theta = self.odom.get_pose()

            rel_x = self.calib.to_m(rel_x_px)
            rel_y = self.calib.to_m(rel_y_px)

            # -----------------------------------------------------
            # COMBINE INITIAL POSE + RELATIVE ECC ODOM
            # -----------------------------------------------------
            global_x = self.init_x + rel_x
            global_y = self.init_y + rel_y
            global_theta = self.init_theta + rel_theta
            global_theta = (global_theta + np.pi) % (2 * np.pi) - np.pi

            self.publish_odom(stamp, global_x, global_y, global_theta)

            proc_msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
            proc_msg.header.stamp = stamp
            self.proc_img_pub.publish(proc_msg)

            self.prev = img

    # ---------------------------------------------------------
    # PUBLISH TRACKED ODOM
    # ---------------------------------------------------------
    def publish_odom(self, stamp, x, y, theta):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_footprint2"

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y

        odom.pose.pose.orientation.z = np.sin(theta / 2)
        odom.pose.pose.orientation.w = np.cos(theta / 2)

        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "base_footprint2"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = np.sin(theta / 2)
        t.transform.rotation.w = np.cos(theta / 2)

        self.tf_broadcaster.sendTransform(t)


# ============================================================
# MAIN
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = SE2OdomTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
