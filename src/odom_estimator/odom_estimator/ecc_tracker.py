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
import time


# ============================================================
# ECC ESTIMATOR (SE2)
# ============================================================
class ECCTracker:
    def __init__(self):
        self.warp_mode = cv2.MOTION_EUCLIDEAN
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            1e-5
        )

    def estimate(self, prev, curr):
        h, w = prev.shape
        cx, cy = w / 2.0, h / 2.0

        warp = np.eye(2, 3, dtype=np.float32)

        # Gaussian mask
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        mask = np.exp(-(xv**2 + yv**2) / 0.25)
        mask = (mask * 255).astype(np.uint8)

        try:
            score, warp = cv2.findTransformECC(
                prev,
                curr,
                warp,
                self.warp_mode,
                self.criteria,
                mask,
                5
            )
        except cv2.error:
            return None

        dx = warp[0, 2]
        dy = warp[1, 2]
        theta = np.arctan2(warp[1, 0], warp[0, 0])

        # rotation compensation
        dx_corr = dx - (cx * (1 - np.cos(theta)) + cy * np.sin(theta))
        dy_corr = dy - (cy * (1 - np.cos(theta)) - cx * np.sin(theta))

        return dx_corr, dy_corr, theta, score


# ============================================================
# POSE INTEGRATOR
# ============================================================
class PoseIntegrator:
    def __init__(self, direction="forward_y"):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # direction ∈ {"forward_x", "reverse_x", "forward_y", "reverse_y"}
        self.direction = direction

    def remap(self, dx, dy):
        if self.direction == "forward_x":
            return dx, dy
        elif self.direction == "reverse_x":
            return -dx, -dy
        elif self.direction == "forward_y":
            return dy, -dx
        elif self.direction == "reverse_y":
            return -dy, dx
        else:
            return dx, dy

    def update(self, dx, dy, dtheta):
        dx, dy = self.remap(dx, dy)

        gx = dx * np.cos(self.theta) - dy * np.sin(self.theta)
        gy = dx * np.sin(self.theta) + dy * np.cos(self.theta)

        self.x += gx
        self.y += gy

        self.theta += dtheta
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get(self):
        return self.x, self.y, self.theta


# ============================================================
# PREPROCESSOR
# ============================================================
class Preprocessor:
    def __init__(self):
        self.crop_w = 470
        self.crop_h = 480

    def process(self, img):
        h, w = img.shape
        cx, cy = w // 2, h // 2

        x1 = cx - self.crop_w // 2
        y1 = cy - self.crop_h // 2
        x2 = cx + self.crop_w // 2
        y2 = cy + self.crop_h // 2

        img = img[y1:y2, x1:x2]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=7)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=7)
        grad = cv2.magnitude(sobelx, sobely)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return grad


# ============================================================
# MAIN NODE
# ============================================================
class ECCTrackerNode(Node):
    def __init__(self):
        super().__init__("ecc_tracker")

        self.bridge = CvBridge()
        self.prev = None

        self.ecc = ECCTracker()
        self.integrator = PoseIntegrator()
        self.preproc = Preprocessor()

        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self.cb, 10
        )

        self.pub_odom = self.create_publisher(Odometry, "/ecc/odom", 10)
        self.pub_debug = self.create_publisher(Image, "/ecc/debug_image", 10)

        self.tf = tf2_ros.TransformBroadcaster(self)

        self.lock = threading.Lock()
        self.latest = None
        self.latest_stamp = None

        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

        self.get_logger().info("ECC Tracker Node Started")

    # --------------------------------------------------------
    # FAST CALLBACK
    # --------------------------------------------------------
    def cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        with self.lock:
            self.latest = img
            self.latest_stamp = msg.header.stamp

    # --------------------------------------------------------
    # WORKER LOOP
    # --------------------------------------------------------
    def loop(self):
        while rclpy.ok():
            with self.lock:
                if self.latest is None:
                    continue
                frame = self.latest.copy()
                stamp = self.latest_stamp
                self.latest = None

            proc = self.preproc.process(frame)
            proc_small = cv2.resize(proc, None, fx=0.5, fy=0.5)

            if self.prev is None:
                self.prev = proc_small
                continue

            result = self.ecc.estimate(self.prev, proc_small)

            if result is None:
                self.warn("ECC failed — skipping frame")
                continue

            dx, dy, dtheta, score = result

            # ------------------------------------------------
            # DEBUGGING OUTPUTS
            # ------------------------------------------------
            if score < 0.3:
                self.warn(f"LOW SCORE {score:.2f} → motion unreliable")
            elif score < 0.5:
                self.info(f"MEDIUM SCORE {score:.2f}")
            else:
                self.info(f"GOOD SCORE {score:.2f}")

            if abs(dx) > 40 or abs(dy) > 40:
                self.warn(f"UNREALISTIC JUMP dx={dx:.1f} dy={dy:.1f}")
                continue

            dx *= 2.0
            dy *= 2.0

            self.integrator.update(dx, dy, dtheta)
            x, y, th = self.integrator.get()

            self.publish(stamp, x, y, th)

            debug_msg = self.bridge.cv2_to_imgmsg(proc_small, "mono8")
            debug_msg.header.stamp = stamp
            self.pub_debug.publish(debug_msg)

            self.prev = proc_small

    # --------------------------------------------------------
    # PUBLISH ODOM + TF
    # --------------------------------------------------------
    def publish(self, stamp, x, y, th):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        odom.pose.pose.position.x = x / 2175.0
        odom.pose.pose.position.y = y / 2175.0

        odom.pose.pose.orientation.z = np.sin(th / 2)
        odom.pose.pose.orientation.w = np.cos(th / 2)

        self.pub_odom.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = x / 2175.0
        t.transform.translation.y = y / 2175.0
        t.transform.rotation.z = np.sin(th / 2)
        t.transform.rotation.w = np.cos(th / 2)

        self.tf.sendTransform(t)

    # --------------------------------------------------------
    # TERMINAL LOG HELPERS
    # --------------------------------------------------------
    def info(self, msg):
        self.get_logger().info(f"[ECC] {msg}")

    def warn(self, msg):
        self.get_logger().warn(f"[ECC WARNING] {msg}")


def main(args=None):
    rclpy.init(args=args)
    node = ECCTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
