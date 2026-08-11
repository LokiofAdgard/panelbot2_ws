import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class InitialOdomPublisher(Node):
    def __init__(self):
        super().__init__('initial_odom_publisher')

        self.pub = self.create_publisher(Odometry, 'initial_odom', 10)

        # Timer fires once after 0.5 seconds
        self.timer = self.create_timer(0.5, self.publish_and_exit)

        # Preset initial pose (modify later when AprilTag is integrated)
        self.initial_x = 0.0
        self.initial_y = 0.0
        self.initial_yaw = 0.0

    def publish_and_exit(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = self.initial_x
        msg.pose.pose.position.y = self.initial_y
        msg.pose.pose.position.z = 0.0

        # Yaw → quaternion
        from math import sin, cos
        qz = sin(self.initial_yaw / 2.0)
        qw = cos(self.initial_yaw / 2.0)

        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        self.pub.publish(msg)
        self.get_logger().info('Initial odometry published. Shutting down.')

        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = InitialOdomPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
