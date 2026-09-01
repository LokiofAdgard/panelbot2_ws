import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from mros_interfaces.msg import MotorCmd
import math
import time

from tf2_ros import Buffer, TransformListener


# -----------------------------
# PID Controller
# -----------------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# -----------------------------
# Square Driver Node
# -----------------------------
class SquareDriverTF(Node):
    def __init__(self):
        super().__init__("square_driver_tf")

        # Publishers
        self.pub_motor = self.create_publisher(MotorCmd, "/motorCmd", 10)
        self.pub_cmdvel = self.create_publisher(Twist, "/cmd_vel", 10)

        # Toggle cmd_vel publishing
        self.use_cmd_vel = True

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer (5 Hz)
        self.dt = 0.2
        self.timer = self.create_timer(self.dt, self.control_loop)

        # State machine
        self.state = "STRAIGHT"
        self.segment_index = 0
        self.target_distance = 1.5
        self.target_turn = math.radians(90)

        # PID controllers
        self.pid_heading = PID(1.2, 0.0, 0.15)
        self.pid_turn = PID(1.0, 0.0, 0.1)

        # Start pose
        self.start_x = None
        self.start_y = None
        self.start_yaw = None

        self.get_logger().info("TF Square Driver with cmd_vel option ready")


    # -----------------------------
    # TF Pose Lookup
    # -----------------------------
    def get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "odom", "base_footprint", rclpy.time.Time()
            )
        except Exception:
            return None

        x = t.transform.translation.x
        y = t.transform.translation.y

        q = t.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return x, y, yaw


    # -----------------------------
    # Publish both motorCmd + cmd_vel
    # -----------------------------
    def publish_commands(self, left, right):
        # MotorCmd
        mc = MotorCmd()
        mc.left_lin = float(left)
        mc.right_lin = float(right)
        self.pub_motor.publish(mc)

        # cmd_vel (optional)
        if self.use_cmd_vel:
            tw = Twist()
            tw.linear.x = (left + right) / 2.0
            tw.angular.z = (right - left) / 0.20   # wheelbase approx
            self.pub_cmdvel.publish(tw)


    # -----------------------------
    # Main Control Loop
    # -----------------------------
    def control_loop(self):
        pose = self.get_pose()
        if pose is None:
            return

        x, y, yaw = pose

        # Initialize segment start
        if self.start_x is None:
            self.start_x = x
            self.start_y = y
            self.start_yaw = yaw

        # -----------------------------
        # STRAIGHT
        # -----------------------------
        if self.state == "STRAIGHT":
            dist = math.sqrt((x - self.start_x)**2 + (y - self.start_y)**2)

            if dist < self.target_distance:
                yaw_error = self.start_yaw - yaw
                correction = self.pid_heading.compute(yaw_error, self.dt)

                base_speed = 0.30
                left = base_speed - correction
                right = base_speed + correction

                self.publish_commands(left, right)

            else:
                self.publish_commands(0.0, 0.0)
                self.state = "TURN"
                self.start_yaw = yaw
                time.sleep(0.3)

        # -----------------------------
        # TURN 90° LEFT
        # -----------------------------
        elif self.state == "TURN":
            yaw_error = yaw - self.start_yaw
            target = self.target_turn

            if abs(yaw_error) < target:
                turn_cmd = self.pid_turn.compute(target - abs(yaw_error), self.dt)
                turn_cmd = max(min(turn_cmd, 0.30), 0.15)

                left = -turn_cmd
                right = turn_cmd

                self.publish_commands(left, right)

            else:
                self.publish_commands(0.0, 0.0)

                self.segment_index += 1
                if self.segment_index >= 4:
                    self.state = "DONE"
                else:
                    self.state = "STRAIGHT"
                    self.start_x = x
                    self.start_y = y
                    self.start_yaw = yaw

                time.sleep(0.3)

        # -----------------------------
        # DONE
        # -----------------------------
        elif self.state == "DONE":
            self.publish_commands(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = SquareDriverTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
