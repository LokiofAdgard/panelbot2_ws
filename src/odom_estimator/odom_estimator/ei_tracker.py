import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from mros_interfaces.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf2_ros
import threading
import time


# ============================================================
# ECC ESTIMATOR (unchanged)
# ============================================================
class ECCEstimator:
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
        mask = np.exp(-(xv**2 + yv**2) / 0.05)
        mask = (mask * 255).astype(np.uint8)
        # mask = None

        try:
            score, warp = cv2.findTransformECC(
                prev, curr, warp,
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

class ORBEstimator:
    def __init__(self, n_features=500, match_ratio=0.8,
                 ransac_thresh_px=4.0, min_inliers=8,
                 fast_threshold=7, edge_threshold=15, n_levels=8):

        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scoreType=cv2.ORB_FAST_SCORE,
            fastThreshold=fast_threshold,
            edgeThreshold=edge_threshold,
            nlevels=n_levels,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.match_ratio = match_ratio
        self.ransac_thresh_px = ransac_thresh_px
        self.min_inliers = min_inliers

        self.last_inlier_pts_curr = np.empty((0, 2), dtype=np.float32)

    def estimate(self, prev, curr):
        self.last_inlier_pts_curr = np.empty((0, 2), dtype=np.float32)

        h, w = prev.shape
        cx, cy = w / 2.0, h / 2.0

        kp1, des1 = self.orb.detectAndCompute(prev, None)
        kp2, des2 = self.orb.detectAndCompute(curr, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return None

        knn = self.bf.knnMatch(des1, des2, k=2)

        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.match_ratio * n.distance:
                good.append(m)

        if len(good) < self.min_inliers:
            return None

        pts_prev = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_curr = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        m_affine, inliers = cv2.estimateAffinePartial2D(
            pts_prev, pts_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh_px
        )

        if m_affine is None or inliers is None:
            return None

        inlier_count = int(inliers.sum())
        if inlier_count < self.min_inliers:
            return None

        inlier_mask = inliers.ravel().astype(bool)
        self.last_inlier_pts_curr = pts_curr[inlier_mask].reshape(-1, 2)

        a, b = m_affine[0, 0], m_affine[0, 1]
        theta = np.arctan2(b, a)
        dx = m_affine[0, 2]
        dy = m_affine[1, 2]

        dx_corr = dx - (cx * (1 - np.cos(theta)) + cy * np.sin(theta))
        dy_corr = dy - (cy * (1 - np.cos(theta)) - cx * np.sin(theta))

        score = inlier_count / len(good)

        return dx_corr, dy_corr, theta, score


# ============================================================
# POSE INTEGRATOR (with direction logic)
# ============================================================
class PoseIntegrator:
    def __init__(self, direction="reverse_x"):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # direction ∈ {"forward_x", "reverse_x", "forward_y", "reverse_y"}
        self.direction = direction

    def remap(self, dx, dy):
        if self.direction == "forward_x":
            return dx, dy
        elif self.direction == "reverse_x":
            return dx, -dy
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

    def set_theta(self, theta):
        self.theta = (theta + np.pi) % (2 * np.pi) - np.pi

    def get(self):
        return self.x, self.y, self.theta


# ============================================================
# PREPROCESSOR
# ============================================================
class Preprocessor:

    def __init__(self, node):
        self.process_mode = node.get_parameter("process_mode").value
        self.crop_w = 470
        self.crop_h = 480

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_soft = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

    def _crop_center(self, img):
        h, w = img.shape
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - self.crop_w // 2)
        y1 = max(0, cy - self.crop_h // 2)
        x2 = min(w, cx + self.crop_w // 2)
        y2 = min(h, cy + self.crop_h // 2)
        return img[y1:y2, x1:x2]

    def process(self, img):
        if self.process_mode == "default":
            return img

        if self.process_mode == "glare_suppress":
            _, bright_mask = cv2.threshold(img, 235, 255, cv2.THRESH_BINARY)
            bright_mask = cv2.dilate(bright_mask, np.ones((5, 5), np.uint8))
            img = cv2.inpaint(img, bright_mask, 3, cv2.INPAINT_TELEA)
            img = self.clahe_soft.apply(img)
            return img

        if self.process_mode == "periodicity_dampen":
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
            img = self.clahe_soft.apply(img)
            return img

        raise ValueError(f"Unknown process_mode: {self.process_mode!r}")


# ============================================================
# MAIN NODE — ECC/ORB + IMU WEIGHTED YAW
# ============================================================
class ECCIMUNode(Node):
    def __init__(self):
        super().__init__("ecc_imu_tracker")

        # Tunable parameters
        self.declare_parameter("yaw_weight_ecc", 0.0)
        self.declare_parameter("yaw_weight_imu", 1.0)
        self.declare_parameter("imu_yaw_smoothing", 0.1)
        self.declare_parameter("ecc_score_min", 0.5)
        self.declare_parameter("direction", "reverse_x")
        self.declare_parameter("process_mode", "default")
        self.declare_parameter("motion_deadband_px", 0.15)

        # --- CHANGE: choose the motion estimator. "ecc" (default,
        # unchanged behavior) or "orb".
        self.declare_parameter("motion_estimator", "orb")

        self.declare_parameter("orb_n_features", 500)
        self.declare_parameter("orb_match_ratio", 0.8)
        self.declare_parameter("orb_ransac_thresh_px", 4.0)
        self.declare_parameter("orb_min_inliers", 8)
        self.declare_parameter("orb_fast_threshold", 7)
        self.declare_parameter("orb_edge_threshold", 15)

        self.preproc = Preprocessor(self)

        self.w_ecc = self.get_parameter("yaw_weight_ecc").value
        self.w_imu = self.get_parameter("yaw_weight_imu").value
        self.imu_smooth = self.get_parameter("imu_yaw_smoothing").value
        self.ecc_score_min = self.get_parameter("ecc_score_min").value
        self.deadband = self.get_parameter("motion_deadband_px").value
        direction = self.get_parameter("direction").value

        motion_estimator = self.get_parameter("motion_estimator").value
        self.motion_estimator_name = motion_estimator
        if motion_estimator == "orb":
            self.estimator = ORBEstimator(
                n_features=self.get_parameter("orb_n_features").value,
                match_ratio=self.get_parameter("orb_match_ratio").value,
                ransac_thresh_px=self.get_parameter("orb_ransac_thresh_px").value,
                min_inliers=self.get_parameter("orb_min_inliers").value,
                fast_threshold=self.get_parameter("orb_fast_threshold").value,
                edge_threshold=self.get_parameter("orb_edge_threshold").value,
            )
        elif motion_estimator == "ecc":
            self.estimator = ECCEstimator()
        else:
            raise ValueError(f"Unknown motion_estimator: {motion_estimator!r}")

        self.score_sum = 0.0
        self.score_count = 0

        # IMU yaw state
        self.imu_yaw = 0.0

        # ROS interfaces
        self.bridge = CvBridge()
        self.prev = None

        self.integrator = PoseIntegrator(direction=direction)

        self.sub_cam = self.create_subscription(Image, "/image_raw", self.cb_cam, 10)
        self.sub_imu = self.create_subscription(Imu, "/imu/data", self.cb_imu, 50)

        self.pub_odom = self.create_publisher(Odometry, "/ei/odom", 10)
        self.pub_debug = self.create_publisher(Image, "/ei/debug_image", 10)
        self.tf = tf2_ros.TransformBroadcaster(self)

        # Threading
        self.cv = threading.Condition()
        self.latest = None
        self.latest_stamp = None

        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            f"ECC/ORB + IMU Hybrid Tracker Started (estimator={motion_estimator})"
        )

    # --------------------------------------------------------
    # IMU CALLBACK — extract yaw from quaternion
    # --------------------------------------------------------
    def cb_imu(self, msg: Imu):
        qx = msg.quaternion.x
        qy = msg.quaternion.y
        qz = msg.quaternion.z
        qw = msg.quaternion.w

        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        delta = (yaw - self.imu_yaw + np.pi) % (2 * np.pi) - np.pi
        self.imu_yaw = self.imu_yaw + self.imu_smooth * delta
        self.imu_yaw = (self.imu_yaw + np.pi) % (2 * np.pi) - np.pi

    # --------------------------------------------------------
    # CAMERA CALLBACK
    # --------------------------------------------------------
    def cb_cam(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        with self.cv:
            self.latest = img
            self.latest_stamp = msg.header.stamp
            self.cv.notify()

    # --------------------------------------------------------
    # WORKER LOOP
    # --------------------------------------------------------
    def loop(self):
        while rclpy.ok():
            with self.cv:
                while self.latest is None and rclpy.ok():
                    self.cv.wait(timeout=0.5)
                if self.latest is None:
                    continue
                frame = self.latest.copy()
                stamp = self.latest_stamp
                self.latest = None

            proc = self.preproc.process(frame)
            img = cv2.resize(proc, None, fx=0.5, fy=0.5)

            if self.prev is None:
                self.publish_debug_image(stamp, img)
                self.prev = img
                continue

            result = self.estimator.estimate(self.prev, img)

            if self.motion_estimator_name == "orb":
                debug_img = self.draw_orb_features(
                    img, self.estimator.last_inlier_pts_curr
                )
            else:
                debug_img = img
            self.publish_debug_image(stamp, debug_img)

            if result is None:
                self.prev = img
                continue

            dx, dy, theta_est, score = result

            self.score_sum += score
            self.score_count += 1
            avg_score = self.score_sum / self.score_count

            if score < self.ecc_score_min:
                self.get_logger().warn(f"Motion score low: {score:.2f}, skipping frame")
                self.prev = img
                continue
            elif score < 0.75:
                self.get_logger().info(f"[MOTION] MEDIUM SCORE {score:.2f}")

            if abs(dx) < self.deadband and abs(dy) < self.deadband:
                dx, dy = 0.0, 0.0

            dx *= 2.0
            dy *= 2.0

            # ------------------------------------------------
            # FUSED YAW (visual estimator + IMU)
            # ------------------------------------------------
            theta_fused = self.w_ecc * theta_est + self.w_imu * self.imu_yaw

            self.integrator.update(dx, dy, 0.0)
            self.integrator.set_theta(theta_fused)

            x, y, th = self.integrator.get()

            self.publish(stamp, x, y, th)

            self.prev = img

    # --------------------------------------------------------
    # PUBLISH FUSED ESTIMATOR + IMU
    # --------------------------------------------------------
    def publish(self, stamp, x, y, th):
        px_m_calib = (650.0 * 9.223 * 1.024)
        # px_m_calib = 2175.0

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        odom.pose.pose.position.x = x / px_m_calib
        odom.pose.pose.position.y = y / px_m_calib

        odom.pose.pose.orientation.z = np.sin(th / 2)
        odom.pose.pose.orientation.w = np.cos(th / 2)

        self.pub_odom.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = x / px_m_calib
        t.transform.translation.y = y / px_m_calib
        t.transform.rotation.z = np.sin(th / 2)
        t.transform.rotation.w = np.cos(th / 2)

        self.tf.sendTransform(t)

    def draw_orb_features(self, img, pts):
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if pts is None or len(pts) == 0:
            return color
        for x, y in pts:
            cv2.circle(
                color,
                (int(round(x)), int(round(y))),
                3,
                (0, 255, 0),
                1,
                lineType=cv2.LINE_AA,
            )
        return color

    def publish_debug_image(self, stamp, img):
        encoding = "bgr8" if img.ndim == 3 else "mono8"
        msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_debug"
        self.pub_debug.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ECCIMUNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()