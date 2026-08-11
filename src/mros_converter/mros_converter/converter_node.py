import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import Float32MultiArray, UInt16MultiArray, Int16MultiArray, Int32MultiArray
from mros_interfaces.msg import Encs, Imu, McStat, PcStat, Pbus, Tof, Ina, MotorCmd


class ConverterNode(Node):
    def __init__(self):
        super().__init__('converter_node')

        # -----------------------------
        # Firmware constants
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
        # Publishers
        # -----------------------------
        self.pub_imu = self.create_publisher(Imu, "imu/data", 10)
        self.pub_tof = self.create_publisher(Tof, "tof/data", 10)

        self.pub_pcstat = self.create_publisher(PcStat, "pc/status", 10)
        self.pub_pbus = self.create_publisher(Pbus, "pc/pbus", 10)

        self.pub_encs = self.create_publisher(Encs, "mc/encs", 10)
        self.pub_mcstat = self.create_publisher(McStat, "mc/status", 10)

        self.pub_mCmd = self.create_publisher(Int16MultiArray, "raw/cmd_vel", 10)

        self.get_logger().info("ConverterNode started — Mros conversion active")

    # ------------------------------------------------------------
    # IMU Converter → Imu.msg
    # ------------------------------------------------------------
    def imu_callback(self, msg: Float32MultiArray):
        raw = msg.data
        if len(raw) < 11:
            self.get_logger().warn("IMU raw data too short")
            return

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
        if len(raw) < 20:
            self.get_logger().warn("PC raw data too short")
            return

        def voltage(v): return float(np.int16(v)) * self.VOLTAGE_LSB
        def current(i): return float(np.int16(i)) * self.CURRENT_LSB
        def power(p):   return float(np.int16(p)) * (25.0 * (self.CURRENT_LSB / 1000.0))
        def temp(t):    return float(np.int16(t)) * self.TEMP_LSB

        # Build INA blocks dynamically
        ina_blocks = []
        for i in range(0, 18, 3):
            ina = Ina()
            ina.voltage = voltage(raw[i])
            ina.current = current(raw[i+1])
            ina.power = power(raw[i+2])
            ina_blocks.append(ina)

        pbus = Pbus()
        pbus.solar, pbus.mppt, pbus.battery, pbus.v5, pbus.v12a, pbus.v12b = ina_blocks
        self.pub_pbus.publish(pbus)

        # PcStat
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
        if len(raw) < 5:
            self.get_logger().warn("MC raw data too short")
            return

        encs = Encs()
        encs.m1, encs.m2, encs.m3, encs.m4 = raw[0:4]
        self.pub_encs.publish(encs)

        bits = raw[4]
        mcstat = McStat()
        mcstat.mode = bits & 0x03
        mcstat.req_stat = bool(bits & (1 << 2))
        mcstat.req_enc = bool(bits & (1 << 3))
        mcstat.req_all = bool(bits & (1 << 4))
        mcstat.req = (bits >> 5) & 0x03

        mc_temp_raw = (raw[4] >> 16) & 0xFFFF
        mcstat.temperature = float(mc_temp_raw) * self.TEMP_LSB

        self.pub_mcstat.publish(mcstat)

    # ------------------------------------------------------------
    # TOF Converter → Tof.msg
    # ------------------------------------------------------------
    def tof_callback(self, msg: UInt16MultiArray):
        raw = msg.data
        if len(raw) < 67:
            self.get_logger().warn("TOF raw data too short")
            return

        out = Tof()
        out.mtof_right = [float(v) for v in raw[0:64]]
        out.tof_front  = float(raw[64] - 60)
        out.tof_left   = float(raw[65] - 60)
        out.tof_back   = float(raw[66] - 60)

        self.pub_tof.publish(out)

    # ------------------------------------------------------------
    # MotorCmd Converter
    # ------------------------------------------------------------
    def mCmd_callback(self, msg: MotorCmd):
        dead_zone = 10
        max_val = 400
        calib = 1

        def scale(val):
            if abs(val) <= dead_zone:
                return 0
            return int(max(-max_val, min(max_val, val * calib)))

        arr = Int16MultiArray()
        arr.data = [scale(msg.left_lin), scale(msg.right_lin)]
        self.pub_mCmd.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ConverterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
