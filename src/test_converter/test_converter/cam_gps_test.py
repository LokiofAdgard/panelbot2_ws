import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, TimeReference, Image, CameraInfo
from geometry_msgs.msg import TwistStamped
from builtin_interfaces.msg import Time
import numpy as np
import math
import cv2

class FakeGPSCam(Node):
    def __init__(self):
        super().__init__('fake_gps_cam')

        # Publishers
        self.pub_fix = self.create_publisher(NavSatFix, '/fix', 10)
        self.pub_vel = self.create_publisher(TwistStamped, '/vel', 10)
        self.pub_time = self.create_publisher(TimeReference, '/time_reference', 10)
        self.pub_img = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera_info', 10)

        # Initial GPS position (change if needed)
        self.lat = 25.2048
        self.lon = 55.2708
        self.alt = 0.0

        # Motion model
        self.speed_mps = 0.05
        self.heading_deg = 0.0

        # Lat/lon conversion
        self.lat_per_meter = 1.0 / 111111.0
        self.lon_per_meter = 1.0 / (111111.0 * abs(math.cos(math.radians(self.lat))))

        # Camera parameters
        self.width = 1280
        self.height = 720
        self.frame_id = "camera_link"
        self.t = 0

        self.timer = self.create_timer(0.1, self.update)  # 10 Hz

    def update(self):
        # --- GPS SIMULATION ---
        dt = 0.1
        distance = self.speed_mps * dt

        # Convert heading to radians
        heading_rad = math.radians(self.heading_deg)

        # Compute ENU displacement
        dx = distance * math.cos(heading_rad)   # East
        dy = distance * math.sin(heading_rad)   # North

        # Convert meters → degrees
        self.lat += dy * self.lat_per_meter
        self.lon += dx * self.lon_per_meter

        stamp = self.get_clock().now().to_msg()

        # NavSatFix
        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = 'gps_link'
        fix.latitude = self.lat
        fix.longitude = self.lon
        fix.altitude = self.alt
        fix.status.status = 0
        fix.status.service = 1
        self.pub_fix.publish(fix)

        # Velocity (ENU projected)
        vel = TwistStamped()
        vel.header.stamp = stamp
        vel.header.frame_id = 'gps_link'
        vel.twist.linear.x = self.speed_mps * math.cos(heading_rad)
        vel.twist.linear.y = self.speed_mps * math.sin(heading_rad)
        self.pub_vel.publish(vel)

        # Time reference
        tref = TimeReference()
        tref.header.stamp = stamp
        tref.time_ref = stamp
        tref.source = 'fake_gps'
        self.pub_time.publish(tref)

        # --- CAMERA SIMULATION ---
        # Create synthetic image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (5, 5, 5)  # bluish background

        # Moving stripe to simulate motion
        stripe_x1 = int((self.t % self.width))
        stripe_x2 = int(((self.t + 800) % self.width))
        stripe_x3 = int(((self.t + 400) % self.width))
        cv2.line(img, (stripe_x1, 0), (stripe_x1 + 200, self.height), (50, 50, 50), 20)
        cv2.line(img, (stripe_x2, 0), (stripe_x2 + 200, self.height), (50, 50, 50), 20)
        cv2.line(img, (stripe_x3, 0), (stripe_x3 + 200, self.height), (50, 50, 50), 20)
        cv2.line(img, (0, 500), (self.width, 500), (50, 50, 50), 20)
        self.t += 10

        # Convert to ROS Image
        ros_img = Image()
        ros_img.header.stamp = stamp
        ros_img.header.frame_id = self.frame_id
        ros_img.height = self.height
        ros_img.width = self.width
        ros_img.encoding = "bgr8"
        ros_img.is_bigendian = 0
        ros_img.step = self.width * 3
        ros_img.data = img.tobytes()
        self.pub_img.publish(ros_img)

        # CameraInfo
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width = self.width
        info.height = self.height
        info.k = [700, 0, self.width/2,
                  0, 700, self.height/2,
                  0, 0, 1]
        info.p = [700, 0, self.width/2, 0,
                  0, 700, self.height/2, 0,
                  0, 0, 1, 0]
        self.pub_info.publish(info)


def main(args=None):
    rclpy.init(args=args)
    node = FakeGPSCam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
