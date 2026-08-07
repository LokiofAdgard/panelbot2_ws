#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import (
    UInt16MultiArray,
    Int32MultiArray,
    Float32MultiArray,
    Float32,
    Int32
)
from geometry_msgs.msg import Quaternion, Vector3


class RawConverter(Node):
    def __init__(self):
        super().__init__('naw_converter')

        # -----------------------------
        # Constants from firmware
        # -----------------------------
        self.CURRENT_LSB = 0.4          # mA
        self.VOLTAGE_LSB = 1.25e-3      # Example: INA219 default (1.25mV)
        self.TEMP_LSB = 0.0625          # °C per bit

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.pc_sub = self.create_subscription(
            UInt16MultiArray, 'raw/pc', self.pc_callback, 10)

        self.mc_sub = self.create_subscription(
            Int32MultiArray, 'raw/mc', self.mc_callback, 10)

        self.imu_sub = self.create_subscription(
            Float32MultiArray, 'raw/imu', self.imu_callback, 10)

        # -----------------------------
        # IMU Publishers
        # -----------------------------
        self.pub_quat = self.create_publisher(Quaternion, 'imu/quaternion', 10)
        self.pub_accel = self.create_publisher(Vector3, 'imu/accel', 10)
        self.pub_gyro = self.create_publisher(Vector3, 'imu/gyro', 10)
        self.pub_mag = self.create_publisher(Float32, 'imu/mag_cal', 10)

        # -----------------------------
        # PC Publishers (19 values)
        # -----------------------------
        def fp(topic): return self.create_publisher(Float32, topic, 10)

        self.pc_pub = {
            "solar_voltage": fp("pc/solar/voltage"),
            "solar_current": fp("pc/solar/current"),
            "solar_power": fp("pc/solar/power"),

            "mppt_voltage": fp("pc/mppt/voltage"),
            "mppt_current": fp("pc/mppt/current"),
            "mppt_power": fp("pc/mppt/power"),

            "bat_voltage": fp("pc/battery/voltage"),
            "bat_current": fp("pc/battery/current"),
            "bat_power": fp("pc/battery/power"),

            "v5_voltage": fp("pc/v5/voltage"),
            "v5_current": fp("pc/v5/current"),
            "v5_power": fp("pc/v5/power"),

            "v12a_voltage": fp("pc/v12a/voltage"),
            "v12a_current": fp("pc/v12a/current"),
            "v12a_power": fp("pc/v12a/power"),

            "v12b_voltage": fp("pc/v12b/voltage"),
            "v12b_current": fp("pc/v12b/current"),
            "v12b_power": fp("pc/v12b/power"),

            "temp": fp("pc/temp")
        }

        # -----------------------------
        # MC Publishers (5 values)
        # -----------------------------
        self.mc_pub_enc1 = self.create_publisher(Int32, "mc/enc_m1", 10)
        self.mc_pub_enc2 = self.create_publisher(Int32, "mc/enc_m2", 10)
        self.mc_pub_enc3 = self.create_publisher(Int32, "mc/enc_m3", 10)
        self.mc_pub_enc4 = self.create_publisher(Int32, "mc/enc_m4", 10)
        self.mc_pub_sta = self.create_publisher(Int32, "mc/sta", 10)
        self.mc_pub_temp = self.create_publisher(Float32, "mc/temp", 10)

        self.get_logger().info("rawConverter started — IMU, PC, MC demux + conversions active")

    # ------------------------------------------------------------
    # IMU Converter (11 values)
    # ------------------------------------------------------------
    def imu_callback(self, msg: Float32MultiArray):
        raw = msg.data

        quat = Quaternion()
        quat.w, quat.x, quat.y, quat.z = raw[0:4]

        accel = Vector3()
        accel.x, accel.y, accel.z = raw[4:7]

        gyro = Vector3()
        gyro.x, gyro.y, gyro.z = raw[7:10]

        mag = Float32()
        mag.data = raw[10]

        self.pub_quat.publish(quat)
        self.pub_accel.publish(accel)
        self.pub_gyro.publish(gyro)
        self.pub_mag.publish(mag)

    # ------------------------------------------------------------
    # PC Converter (19 values)
    # ------------------------------------------------------------
    def pc_callback(self, msg: UInt16MultiArray):
        raw = msg.data

        # Helper conversion functions
        def voltage(v_reg):
            return float(v_reg) * self.VOLTAGE_LSB

        def current(i_reg):
            return float(i_reg) * self.CURRENT_LSB  # mA

        def power(p_reg):
            return float(p_reg) * (25.0 * (self.CURRENT_LSB / 1000.0))

        def temp(t_reg):
            return float(t_reg) * self.TEMP_LSB

        # Map raw values to converted values
        converted = [
            voltage(raw[0]), current(raw[1]), power(raw[2]),
            voltage(raw[3]), current(raw[4]), power(raw[5]),
            voltage(raw[6]), current(raw[7]), power(raw[8]),
            voltage(raw[9]), current(raw[10]), power(raw[11]),
            voltage(raw[12]), current(raw[13]), power(raw[14]),
            voltage(raw[15]), current(raw[16]), power(raw[17]),
            temp(raw[18])
        ]

        # Publish each converted value
        for key, value in zip(self.pc_pub.keys(), converted):
            msg_out = Float32()
            msg_out.data = value
            self.pc_pub[key].publish(msg_out)

    # ------------------------------------------------------------
    # MC Converter (5 values)
    # ------------------------------------------------------------
    def mc_callback(self, msg: Int32MultiArray):
        raw = msg.data

        enc1 = Int32(); enc1.data = raw[0]
        enc2 = Int32(); enc2.data = raw[1]
        enc3 = Int32(); enc3.data = raw[2]
        enc4 = Int32(); enc4.data = raw[3]

        sta =  Int32(); sta.data = (raw[4] & 0xFFFF)
        temp = Float32(); temp.data = float(raw[4] >> 16) * self.TEMP_LSB

        self.mc_pub_enc1.publish(enc1)
        self.mc_pub_enc2.publish(enc2)
        self.mc_pub_enc3.publish(enc3)
        self.mc_pub_enc4.publish(enc4)
        self.mc_pub_sta.publish(sta)
        self.mc_pub_temp.publish(temp)


def main(args=None):
    rclpy.init(args=args)
    node = RawConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
