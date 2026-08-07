import rclpy
from rclpy.node import Node
from mros_interfaces.msg import Pbus

class ConvPublisher(Node):
    def __init__(self):
        super().__init__('conv_publisher')
        self.pub = self.create_publisher(Pbus, 'pbus_topic', 10)
        self.timer = self.create_timer(1.0, self.publish_pbus)

    def publish_pbus(self):
        msg = Pbus()
        msg.voltage = 12.0
        msg.current = 2.0
        msg.power = 24.0
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ConvPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
