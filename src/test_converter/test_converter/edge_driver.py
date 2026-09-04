import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

from mros_interfaces.msg import MotorCmd
from mros_interfaces.msg import EdgeData

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

class LowPassFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def reset(self):
        self.value = None

    def update(self, measurement):
        if self.value is None:
            self.value = measurement
        else:
            self.value += self.alpha * (measurement - self.value)

        return self.value


class PID:
    def __init__(
        self,
        kp,
        ki,
        kd,
        output_min=-float("inf"),
        output_max=float("inf"),
        integral_limit=float("inf"),
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit

        self.prev_error = None
        self.integral = 0.0

    def reset(self):
        self.prev_error = None
        self.integral = 0.0

    def compute(self, error, dt):
        if dt <= 0.0:
            return 0.0

        # Integral
        self.integral += error * dt
        self.integral = clamp(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
        )

        # Derivative
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return clamp(
            output,
            self.output_min,
            self.output_max,
        )


# ============================================================
# Right Edge Following Driver
# ============================================================

class RightEdgeDriver(Node):

    def __init__(self):
        super().__init__("right_edge_driver")

        # ----------------------------------------------------
        # Tunable parameters
        # ----------------------------------------------------

        # Target edge geometry
        self.target_right_dist = 60.0
        self.target_right_theta = 0.0

        # Forward speed
        self.base_speed = 0.25

        # Emergency threshold
        self.emergency_dist = 0.0

        # Wheel separation used only for optional cmd_vel
        self.wheelbase = 0.20

        # Control rate
        self.control_dt = 0.05  # 20 Hz

        self.pid_dist = PID(
            kp=0.010,
            ki=0.000,
            kd=0.002,
            output_min=-0.15,
            output_max=0.15,
            integral_limit=20.0,
        )

        self.pid_theta = PID(
            kp=0.015,
            ki=0.000,
            kd=0.002,
            output_min=-0.15,
            output_max=0.15,
            integral_limit=10.0,
        )

        self.max_steering = 0.20

        self.dist_filter = LowPassFilter(alpha=0.25)
        self.theta_filter = LowPassFilter(alpha=0.20)

        self.front_tof = False
        self.left_tof = False
        self.rear_tof = False

        self.raw_right_dist = None
        self.raw_right_theta = None

        self.right_dist = None
        self.right_theta = None

        self.data_received = False

        self.state = "WAITING"

        self.pub_motor = self.create_publisher(
            MotorCmd,
            "/motorCmd",
            10,
        )

        self.pub_cmdvel = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        # Set True if you also want /cmd_vel published
        self.use_cmd_vel = True

        self.sub_edge = self.create_subscription(
            EdgeData,
            "/tof/edge_data",
            self.edge_callback,
            10,
        )

        self.timer = self.create_timer(
            self.control_dt,
            self.control_loop,
        )

        # Debug timer at 5 Hz
        self.debug_timer = self.create_timer(
            0.2,
            self.print_debug,
        )

        self.get_logger().info(
            "Right Edge Driver ready"
        )
        self.get_logger().info(
            f"Target: dist={self.target_right_dist:.1f}, "
            f"theta={self.target_right_theta:.2f}, "
            f"speed={self.base_speed:.2f}"
        )

    def edge_callback(self, msg):

        self.front_tof = msg.front_tof
        self.left_tof = msg.left_tof
        self.rear_tof = msg.rear_tof

        self.raw_right_dist = float(msg.right_dist)
        self.raw_right_theta = float(msg.right_theta)

        # Filter measurements
        self.right_dist = self.dist_filter.update(
            self.raw_right_dist
        )

        self.right_theta = self.theta_filter.update(
            self.raw_right_theta
        )

        self.data_received = True

        # Start automatically after first valid data
        if self.state == "WAITING":
            self.state = "DRIVING"
            self.get_logger().info(
                "Edge data received. Starting edge following."
            )

    def publish_commands(self, left, right):

        # Safety clamp
        left = clamp(left, -1.0, 1.0)
        right = clamp(right, -1.0, 1.0)

        # Motor command
        mc = MotorCmd()
        mc.left_lin = float(left)
        mc.right_lin = float(right)
        self.pub_motor.publish(mc)

        # Optional cmd_vel
        if self.use_cmd_vel:
            tw = Twist()

            tw.linear.x = (left + right) / 2.0

            # Positive angular.z corresponds to left turn.
            tw.angular.z = (
                (right - left) / self.wheelbase
            )

            self.pub_cmdvel.publish(tw)

    def stop_robot(self):
        self.publish_commands(0.0, 0.0)

    def control_loop(self):

        if self.state == "WAITING":
            self.stop_robot()
            return

        if self.state == "DONE":
            self.stop_robot()
            return

        if self.state == "EMERGENCY_STOP":
            self.stop_robot()
            return

        if not self.data_received:
            self.stop_robot()
            return

        if self.front_tof:
            self.stop_robot()

            self.state = "DONE"

            self.pid_dist.reset()
            self.pid_theta.reset()

            self.get_logger().info(
                "Front TOF detected. Task complete."
            )
            return

        if self.right_dist < self.emergency_dist:

            self.stop_robot()

            self.state = "EMERGENCY_STOP"

            self.pid_dist.reset()
            self.pid_theta.reset()

            self.get_logger().error(
                f"EMERGENCY STOP: right_dist="
                f"{self.right_dist:.2f} < "
                f"{self.emergency_dist:.2f}"
            )
            return

        dist_error = (
            self.target_right_dist
            - self.right_dist
        )

        dist_correction = self.pid_dist.compute(
            dist_error,
            self.control_dt,
        )

        theta_error = (
            self.right_theta
            - self.target_right_theta
        )

        theta_correction = self.pid_theta.compute(
            theta_error,
            self.control_dt,
        )

        steering = (
            dist_correction
            - theta_correction
        )

        steering = clamp(
            steering,
            -self.max_steering,
            self.max_steering,
        )

        left = self.base_speed - steering
        right = self.base_speed + steering

        self.publish_commands(left, right)

        self.dist_error = dist_error
        self.theta_error = theta_error
        self.dist_correction = dist_correction
        self.theta_correction = theta_correction
        self.steering = steering
        self.left_cmd = left
        self.right_cmd = right

    def print_debug(self):

        if not self.data_received:
            self.get_logger().info(
                "[EDGE] WAITING FOR DATA"
            )
            return

        if self.state == "WAITING":
            return

        if self.state == "DONE":
            self.get_logger().info(
                "[EDGE] DONE | front_tof=True"
            )
            return

        if self.state == "EMERGENCY_STOP":
            self.get_logger().error(
                f"[EDGE] EMERGENCY STOP | "
                f"dist={self.right_dist:.2f}"
            )
            return

        self.get_logger().info(
            "\n"
            "================ EDGE FOLLOW ================\n"
            f" State      : {self.state}\n"
            f" Right Dist : {self.right_dist:7.2f} "
            f"(raw {self.raw_right_dist:7.2f}) | "
            f"error {self.dist_error:+7.2f}\n"
            f" Right Theta: {self.right_theta:+7.3f} "
            f"(raw {self.raw_right_theta:+7.3f}) | "
            f"error {self.theta_error:+7.3f}\n"
            f" PID Dist   : {self.dist_correction:+7.3f}\n"
            f" PID Theta  : {self.theta_correction:+7.3f}\n"
            f" Steering   : {self.steering:+7.3f}\n"
            f" Motor Cmd  : L={self.left_cmd:+6.3f}  "
            f"R={self.right_cmd:+6.3f}\n"
            f" TOF        : F={self.front_tof} "
            f"L={self.left_tof} R={self.rear_tof}\n"
            "============================================"
        )

def main(args=None):

    rclpy.init(args=args)

    node = RightEdgeDriver()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()