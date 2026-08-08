import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import Float32MultiArray, UInt16MultiArray, Int16MultiArray, Int32MultiArray
from mros_interfaces.msg import (
    Encs,
    Imu,
    McStat,
    PcStat,
    Pbus,
    Tof,
    Ina,
    MotorCmd
)


class RawConverter(Node):
    def __init__(self):
        super().__init__('raw_converter')

        # -----------------------------
        # Constants from firmware
        # -----------------------------
        self.CURRENT_LSB = 0.4          # mA
        self.VOLTAGE_LSB = 1.25e-3      # V
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

        self.tof_sub = self.create_subscription(
            UInt16MultiArray, 'raw/tof', self.tof_callback, 10)

        self.mcmd_sub = self.create_subscription(
            MotorCmd, '/motorCmd', self.mCmd_callback, 10)

        # -----------------------------
        # Publishers (custom messages)
        # -----------------------------
        self.pub_imu = self.create_publisher(Imu, "imu/data", 10)
        self.pub_tof = self.create_publisher(Tof, "tof/data", 10)

        self.pub_pcstat = self.create_publisher(PcStat, "pc/status", 10)
        self.pub_pbus = self.create_publisher(Pbus, "pc/pbus", 10)

        self.pub_encs = self.create_publisher(Encs, "mc/encs", 10)
        self.pub_mcstat = self.create_publisher(McStat, "mc/status", 10)

        self.pub_mCmd = self.create_publisher(Int16MultiArray, "raw/cmd_vel", 10)

        self.get_logger().info("RawConverter started — IMU, PC, MC, TOF conversion active")

    # ------------------------------------------------------------
    # IMU Converter → Imu.msg
    # ------------------------------------------------------------
    def imu_callback(self, msg: Float32MultiArray):
        raw = msg.data

        out = Imu()
        out.quaternion.w, out.quaternion.x, out.quaternion.y, out.quaternion.z = raw[0:4]
        out.accel.x, out.accel.y, out.accel.z = raw[4:7]
        out.gyro.x, out.gyro.y, out.gyro.z = raw[7:10]
        out.calibration.data = raw[10]

        self.pub_imu.publish(out)

    # ------------------------------------------------------------
    # PC Converter → Pbus.msg + PcStat.msg
    # ------------------------------------------------------------
    def pc_callback(self, msg: UInt16MultiArray):
        raw = msg.data

        def to_int16(x): return np.int16(x).item()

        def voltage(v): return float(to_int16(v)) * self.VOLTAGE_LSB
        def current(i): return float(to_int16(i)) * self.CURRENT_LSB
        def power(p):   return float(to_int16(p)) * (25.0 * (self.CURRENT_LSB / 1000.0))
        def temp(t):    return float(to_int16(t)) * self.TEMP_LSB

        # -----------------------------
        # Build INA blocks
        # -----------------------------
        solar = Ina()
        solar.voltage = voltage(raw[0])
        solar.current = current(raw[1])
        solar.power = power(raw[2])

        mppt = Ina()
        mppt.voltage = voltage(raw[3])
        mppt.current = current(raw[4])
        mppt.power = power(raw[5])

        battery = Ina()
        battery.voltage = voltage(raw[6])
        battery.current = current(raw[7])
        battery.power = power(raw[8])

        v5 = Ina()
        v5.voltage = voltage(raw[9])
        v5.current = current(raw[10])
        v5.power = power(raw[11])

        v12a = Ina()
        v12a.voltage = voltage(raw[12])
        v12a.current = current(raw[13])
        v12a.power = power(raw[14])

        v12b = Ina()
        v12b.voltage = voltage(raw[15])
        v12b.current = current(raw[16])
        v12b.power = power(raw[17])

        # -----------------------------
        # Pbus.msg (now contains 6 INA blocks)
        # -----------------------------
        pbus = Pbus()
        pbus.solar = solar
        pbus.mppt = mppt
        pbus.battery = battery
        pbus.v5 = v5
        pbus.v12a = v12a
        pbus.v12b = v12b

        self.pub_pbus.publish(pbus)

        # -----------------------------
        # PcStat.msg (unchanged)
        # -----------------------------
        bits = raw[19]

        stat = PcStat()
        stat.mode = bits & 0x03
        stat.sol = bool(bits & (1 << 2))
        stat.sol_ut = bool(bits & (1 << 3))
        stat.mppt_in = bool(bits & (1 << 4))
        stat.bat = bool(bits & (1 << 5))

        stat.en_v5 = bool(bits & (1 << 6))
        stat.en_v12a = bool(bits & (1 << 7))
        stat.en_v12b = bool(bits & (1 << 8))
        stat.en_fan = bool(bits & (1 << 9))

        stat.req = (bits >> 10) & 0x0F

        stat.temperature = temp(raw[18])

        self.pub_pcstat.publish(stat)


    # ------------------------------------------------------------
    # MC Converter → Encs.msg + McStat.msg
    # ------------------------------------------------------------
    def mc_callback(self, msg: Int32MultiArray):
        raw = msg.data

        # Encoders
        encs = Encs()
        encs.m1 = raw[0]
        encs.m2 = raw[1]
        encs.m3 = raw[2]
        encs.m4 = raw[3]
        self.pub_encs.publish(encs)

        # MC Status
        bits = raw[4]

        mcstat = McStat()
        mcstat.mode = bits & 0x03
        mcstat.req_stat = bool(bits & (1 << 2))
        mcstat.req_enc = bool(bits & (1 << 3))
        mcstat.req_all = bool(bits & (1 << 4))
        mcstat.req = (bits >> 5) & 0x03

        mcstat.temperature = float(raw[4] >> 16) * self.TEMP_LSB

        self.pub_mcstat.publish(mcstat)

    # ------------------------------------------------------------
    # TOF Converter → Tof.msg
    # ------------------------------------------------------------
    def tof_callback(self, msg: Float32MultiArray):
        raw = msg.data

        out = Tof()
        out.mtof_right = [float(v) for v in raw[0:64]]
        out.tof_front  = float(raw[64] - 60)
        out.tof_left   = float(raw[65] - 60)
        out.tof_back   = float(raw[66] - 60)

        self.pub_tof.publish(out)

    # ------------------------------------------------------------
    # MCmd Converter
    # ------------------------------------------------------------
    def mCmd_callback(self, msg: MotorCmd):
        dead_zone = 10
        max_val = 400
        calib = 1
        def scale(val):
            if -dead_zone <= val <= dead_zone:
                return 0

            return int(max(-max_val, min(max_val, val * calib)))

        left = scale(msg.left_lin)
        right = scale(msg.right_lin)

        arr = Int16MultiArray()
        arr.data = [left, right]
        self.pub_mCmd.publish(arr)



def main(args=None):
        rclpy.init(args=args)
        node = RawConverter()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
        main()
