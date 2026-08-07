#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial

from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import TwistStamped
import math

class NMEASerialNode(Node):
    def __init__(self):
        super().__init__('nmea_serial_node')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)

        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baud').get_parameter_value().integer_value

        self.ser = serial.Serial(port, baud, timeout=1)

        self.pub_fix = self.create_publisher(NavSatFix, 'gps/fix', 10)
        self.pub_vel = self.create_publisher(TwistStamped, 'gps/vel', 10)

        self.create_timer(0.01, self.read_serial)

        self.get_logger().info(f"Reading NMEA from {port} @ {baud}")

    def safe_float(self, value, default=0.0):
        """Convert safely to float, return default if empty."""
        try:
            return float(value)
        except:
            return default

    def nmea_to_decimal(self, value, direction):
        if value == "":
            return 0.0
        raw = float(value)
        deg = int(raw / 100)
        minutes = raw - deg * 100
        dec = deg + minutes / 60.0
        if direction in ['S', 'W']:
            dec = -dec
        return dec

    def read_serial(self):
        try:
            line = self.ser.readline().decode('ascii', errors='ignore').strip()
            if not line.startswith('$'):
                return

            parts = line.split(',')

            # -----------------------------
            # GGA: Fix + HDOP + Satellites
            # -----------------------------
            if line.startswith("$GNGGA") or line.startswith("$GPGGA"):
                fix = NavSatFix()
                fix.header.stamp = self.get_clock().now().to_msg()
                fix.header.frame_id = "gps"

                gps_qual = self.safe_float(parts[6], 0)

                if gps_qual == 0:
                    # No fix
                    fix.status.status = NavSatStatus.STATUS_NO_FIX
                    fix.status.service = NavSatStatus.SERVICE_GPS
                    fix.latitude = 0.0
                    fix.longitude = 0.0
                    fix.altitude = 0.0
                    fix.position_covariance = [999,0,0, 0,999,0, 0,0,999]
                    fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
                    self.pub_fix.publish(fix)
                    return

                # Valid fix
                fix.status.status = NavSatStatus.STATUS_FIX
                fix.status.service = NavSatStatus.SERVICE_GPS

                fix.latitude = self.nmea_to_decimal(parts[2], parts[3])
                fix.longitude = self.nmea_to_decimal(parts[4], parts[5])
                fix.altitude = self.safe_float(parts[9], 0.0)

                hdop = self.safe_float(parts[8], 1.0)
                cov = hdop * hdop
                fix.position_covariance = [cov,0,0, 0,cov,0, 0,0,cov]
                fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

                self.pub_fix.publish(fix)

            # -----------------------------
            # RMC: Speed + Heading
            # -----------------------------
            if line.startswith("$GNRMC") or line.startswith("$GPRMC"):
                vel = TwistStamped()
                vel.header.stamp = self.get_clock().now().to_msg()
                vel.header.frame_id = "gps"

                speed_knots = self.safe_float(parts[7], 0.0)
                heading_deg = self.safe_float(parts[8], 0.0)

                speed_mps = speed_knots * 0.514444

                vel.twist.linear.x = speed_mps * math.cos(math.radians(heading_deg))
                vel.twist.linear.y = speed_mps * math.sin(math.radians(heading_deg))
                vel.twist.linear.z = 0.0

                self.pub_vel.publish(vel)

        except Exception as e:
            self.get_logger().warn(str(e))


def main(args=None):
    rclpy.init(args=args)
    node = NMEASerialNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
