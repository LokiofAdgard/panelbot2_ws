import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray, KeyValue

from mros_interfaces.msg import Pbus, PcStat, McStat, Tof
import time


class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("enable_diagnostics", False)
        self.enable_diag = self.get_parameter("enable_diagnostics").get_parameter_value().bool_value

        # -----------------------------
        # Diagnostics publisher
        # -----------------------------
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        # -----------------------------
        # Fault tracking
        # -----------------------------
        self.active_faults = set()

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.create_subscription(Pbus, "pc/pbus", self.pbus_callback, 10)
        self.create_subscription(PcStat, "pc/status", self.pcstat_callback, 10)
        self.create_subscription(McStat, "mc/status", self.mcstat_callback, 10)
        self.create_subscription(Tof, "tof/data", self.tof_callback, 10)

        # -----------------------------
        # Last update timestamps
        # -----------------------------
        self.last_tof_time = time.time()
        self.last_mc_time = time.time()
        self.last_pc_time = time.time()

        # -----------------------------
        # Timers
        # -----------------------------
        self.create_timer(1.0, self.stale_check)
        self.create_timer(5.0, self.print_fault_list)

        self.get_logger().info("SafetyMonitorNode (diagnostics mode) started")

    # ------------------------------------------------------------
    # Utility: publish diagnostic status
    # ------------------------------------------------------------
    def publish_diag(self, name, level, message, kv_pairs=None):
        if not self.enable_diag:
            return

        diag = DiagnosticStatus()
        diag.name = name
        diag.level = level
        diag.message = message

        if kv_pairs:
            diag.values = [KeyValue(key=k, value=str(v)) for k, v in kv_pairs.items()]

        arr = DiagnosticArray()
        arr.status = [diag]
        arr.header.stamp = self.get_clock().now().to_msg()

        self.pub_diag.publish(arr)

    # ------------------------------------------------------------
    # Fault tracking (stateful)
    # ------------------------------------------------------------
    def set_fault(self, fault_name):
        if fault_name not in self.active_faults:
            self.active_faults.add(fault_name)
            self.get_logger().warn(f"[FAULT] {fault_name}")

    def clear_fault(self, fault_name):
        if fault_name in self.active_faults:
            self.active_faults.remove(fault_name)
            self.get_logger().info(f"[CLEARED] {fault_name}")

    # ------------------------------------------------------------
    # Periodic fault list printout
    # ------------------------------------------------------------
    def print_fault_list(self):
        self.get_logger().info("====================================")
        self.get_logger().info(" Active Faults:")
        if not self.active_faults:
            self.get_logger().info("   None")
        else:
            for f in self.active_faults:
                self.get_logger().info(f"   - {f}")
        self.get_logger().info("====================================")

    # ------------------------------------------------------------
    # Pbus checks
    # ------------------------------------------------------------
    def pbus_callback(self, msg: Pbus):
        self.last_pc_time = time.time()

        # V5 rail
        if msg.v5.voltage > 5.2:
            self.set_fault("V5 overvoltage")
            self.publish_diag("PowerBus/V5", DiagnosticStatus.ERROR,
                              "V5 rail overvoltage", {"voltage": msg.v5.voltage})
        else:
            self.clear_fault("V5 overvoltage")
            self.publish_diag("PowerBus/V5", DiagnosticStatus.OK,
                              "V5 rail normal", {"voltage": msg.v5.voltage})

        # V12A rail
        if msg.v12a.voltage > 12.5:
            self.set_fault("V12A overvoltage")
            self.publish_diag("PowerBus/V12A", DiagnosticStatus.ERROR,
                              "V12A rail overvoltage", {"voltage": msg.v12a.voltage})
        else:
            self.clear_fault("V12A overvoltage")
            self.publish_diag("PowerBus/V12A", DiagnosticStatus.OK,
                              "V12A rail normal", {"voltage": msg.v12a.voltage})

        # Battery current
        if msg.battery.current > 3000:
            self.set_fault("Battery current high")
            self.publish_diag("PowerBus/Battery", DiagnosticStatus.WARN,
                              "Battery current high (>3A)", {"current_mA": msg.battery.current})
        else:
            self.clear_fault("Battery current high")
            self.publish_diag("PowerBus/Battery", DiagnosticStatus.OK,
                              "Battery current normal", {"current_mA": msg.battery.current})

    # ------------------------------------------------------------
    # PcStat checks
    # ------------------------------------------------------------
    def pcstat_callback(self, msg: PcStat):
        self.last_pc_time = time.time()

        if msg.temperature > 70:
            self.set_fault("PC overheating")
            self.publish_diag("PowerController", DiagnosticStatus.ERROR,
                              "Power controller overheating", {"temp_C": msg.temperature})
        else:
            self.clear_fault("PC overheating")
            self.publish_diag("PowerController", DiagnosticStatus.OK,
                              "Power controller normal", {"temp_C": msg.temperature})

    # ------------------------------------------------------------
    # McStat checks
    # ------------------------------------------------------------
    def mcstat_callback(self, msg: McStat):
        self.last_mc_time = time.time()

        if msg.temperature > 70:
            self.set_fault("MC overheating")
            self.publish_diag("MotorController", DiagnosticStatus.ERROR,
                              "Motor controller overheating", {"temp_C": msg.temperature})
        else:
            self.clear_fault("MC overheating")
            self.publish_diag("MotorController", DiagnosticStatus.OK,
                              "Motor controller normal", {"temp_C": msg.temperature})

    # ------------------------------------------------------------
    # TOF checks
    # ------------------------------------------------------------
    def tof_callback(self, msg: Tof):
        self.last_tof_time = time.time()

        # Right array offline
        if all(v == 0 for v in msg.mtof_right):
            self.set_fault("TOF right offline")
            self.publish_diag("TOF/Right", DiagnosticStatus.ERROR,
                              "Right TOF array offline")
        else:
            self.clear_fault("TOF right offline")
            self.publish_diag("TOF/Right", DiagnosticStatus.OK,
                              "Right TOF array normal")

        # Individual sensors
        sensors = {
            "Front": msg.tof_front,
            "Left": msg.tof_left,
            "Back": msg.tof_back
        }

        for name, val in sensors.items():
            fault_name = f"TOF {name} offline"

            if val > 20000:
                self.set_fault(fault_name)
                self.publish_diag(f"TOF/{name}", DiagnosticStatus.ERROR,
                                  f"{name} TOF offline", {"value": val})
            elif val < 0:
                self.set_fault(f"TOF {name} invalid negative")
                self.publish_diag(f"TOF/{name}", DiagnosticStatus.ERROR,
                                  f"{name} TOF invalid negative", {"value": val})
            else:
                self.clear_fault(fault_name)
                self.publish_diag(f"TOF/{name}", DiagnosticStatus.OK,
                                  f"{name} TOF normal", {"value": val})

    # ------------------------------------------------------------
    # Stale sensor detection
    # ------------------------------------------------------------
    def stale_check(self):
        now = time.time()

        if now - self.last_tof_time > 2.0:
            self.set_fault("TOF stale")
            self.publish_diag("TOF", DiagnosticStatus.STALE,
                              "TOF sensor not updating")
        else:
            self.clear_fault("TOF stale")

        if now - self.last_mc_time > 2.0:
            self.set_fault("MC stale")
            self.publish_diag("MotorController", DiagnosticStatus.STALE,
                              "Motor controller not updating")
        else:
            self.clear_fault("MC stale")

        if now - self.last_pc_time > 2.0:
            self.set_fault("PC stale")
            self.publish_diag("PowerController", DiagnosticStatus.STALE,
                              "Power controller not updating")
        else:
            self.clear_fault("PC stale")


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
