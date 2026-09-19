import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus


class GPSOdom(Node):

    def __init__(self):
        super().__init__('gps_odom')

        # ============================================================
        # Parameters
        # ============================================================

        # Fixed rotation between GPS ENU frame and /ei/odom frame.
        self.declare_parameter('yaw_offset', -0.76794)
        self.yaw_offset = self.get_parameter(
            'yaw_offset'
        ).get_parameter_value().double_value

        self.declare_parameter('debug', True)
        self.debug = self.get_parameter(
            'debug'
        ).get_parameter_value().bool_value

        self.declare_parameter('debug_period', 1.0)
        self.debug_period = self.get_parameter(
            'debug_period'
        ).get_parameter_value().double_value

        self.declare_parameter('scale_min_step_m', 0.05)
        self.scale_min_step_m = self.get_parameter(
            'scale_min_step_m'
        ).get_parameter_value().double_value

        self.declare_parameter('scale_warmup_pairs', 20)
        self.scale_warmup_pairs = self.get_parameter(
            'scale_warmup_pairs'
        ).get_parameter_value().integer_value

        self.declare_parameter('scale_smoothing_alpha', 0.03)
        self.scale_smoothing_alpha = self.get_parameter(
            'scale_smoothing_alpha'
        ).get_parameter_value().double_value

        self.declare_parameter('process_noise_floor_m', 0.06)
        self.process_noise_floor_m = self.get_parameter(
            'process_noise_floor_m'
        ).get_parameter_value().double_value

        self.declare_parameter('process_noise_per_meter', 0.005)
        self.process_noise_per_meter = self.get_parameter(
            'process_noise_per_meter'
        ).get_parameter_value().double_value

        self.declare_parameter('gps_measurement_noise_std', 1.5)
        self.gps_measurement_noise_std = self.get_parameter(
            'gps_measurement_noise_std'
        ).get_parameter_value().double_value

        self.declare_parameter('kf_initial_variance', 0.01)
        self.kf_initial_variance = self.get_parameter(
            'kf_initial_variance'
        ).get_parameter_value().double_value

        # ============================================================
        # Subscribers
        # ============================================================

        self.odom_sub = self.create_subscription(
            Odometry,
            '/ei/odom',
            self.odom_callback,
            10
        )

        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10
        )

        # ============================================================
        # Publishers
        # ============================================================

        # Kalman-fused GPS + ECC position.
        self.gps_odom_pub = self.create_publisher(
            Odometry,
            '/gps/odom',
            10
        )

        # Bias-corrected-only ECC trajectory (no GPS involved) -- what
        # to dead-reckon from during a GPS dropout.
        self.ecc_scaled_pub = self.create_publisher(
            Odometry,
            '/ei/odom_scaled',
            10
        )

        # ============================================================
        # State
        # ============================================================

        self.latest_odom = None

        # GPS origin -- locked directly from a single fix, matching
        # ECC's own pose at that instant "as is" (no averaging).
        self.origin_set = False
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None

        self.origin_odom_x = None
        self.origin_odom_y = None
        self.origin_odom_z = None

        self.gps_x = None
        self.gps_y = None
        self.gps_z = None

        # Scale estimator state.
        self._scale_sum_xy = 0.0
        self._scale_sum_xx = 0.0
        self._scale_pair_count = 0
        self.ecc_scale_raw = 1.0
        self.ecc_scale = 1.0

        self._last_sync_gps_en = None    # (east, north) at last GPS update
        self._last_sync_odom_xy = None   # (odom_x, odom_y) at last GPS update

        # Bias-corrected-only ECC trajectory (incremental, no GPS).
        self.scaled_x = None
        self.scaled_y = None
        self._last_odom_xy = None        # (odom_x, odom_y) at last ECC msg

        # Kalman filter state (independent per axis).
        self.kf_x = None
        self.kf_y = None
        self.kf_p_x = None
        self.kf_p_y = None

        self.last_debug_time = self.get_clock().now()

        self.EARTH_RADIUS = 6378137.0

        self.get_logger().info('======================================')
        self.get_logger().info(' GPS Odom Node Started (Kalman fusion)')
        self.get_logger().info('======================================')
        self.get_logger().info(
            f'yaw_offset = {math.degrees(self.yaw_offset):.3f} deg '
            f'({self.yaw_offset:.6f} rad)'
        )
        self.get_logger().info('Waiting for /ei/odom and /fix...')

    # ================================================================
    # ODOM CALLBACK -- Kalman PREDICT step + bias-corrected-only track
    # ================================================================

    def odom_callback(self, msg):

        self.latest_odom = msg

        if not self.origin_set:
            return

        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y

        if self._last_odom_xy is None:
            self._last_odom_xy = (odom_x, odom_y)
            return

        ecc_dx = odom_x - self._last_odom_xy[0]
        ecc_dy = odom_y - self._last_odom_xy[1]
        self._last_odom_xy = (odom_x, odom_y)

        # --- Bias-corrected-only trajectory (no GPS). Incremental, so
        # a zero ECC delta contributes exactly zero -- no fluctuation
        # from a changing scale estimate being reapplied to the whole
        # trip, only ever to the newest step.
        self.scaled_x += self.ecc_scale * ecc_dx
        self.scaled_y += self.ecc_scale * ecc_dy

        # --- Kalman PREDICT: bias-corrected ECC delta is the motion
        # model / control input. Uncertainty grows by a fixed floor
        # plus a term proportional to distance moved (residual scale
        # error not fully removed by the corrector above).
        step = math.hypot(ecc_dx, ecc_dy)
        q_std = self.process_noise_floor_m + self.process_noise_per_meter * step
        q = q_std * q_std

        self.kf_x += self.ecc_scale * ecc_dx
        self.kf_y += self.ecc_scale * ecc_dy
        self.kf_p_x += q
        self.kf_p_y += q

    # ================================================================
    # GPS CALLBACK -- origin lock, scale-estimator update, Kalman UPDATE
    # ================================================================

    def gps_callback(self, msg):

        if msg.status.status == NavSatStatus.STATUS_NO_FIX:
            if self.debug:
                self.get_logger().warn('GPS: NO FIX', throttle_duration_sec=2.0)
            return

        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            self.get_logger().warn('GPS contains NaN latitude/longitude')
            return

        if self.latest_odom is None:
            if self.debug:
                self.get_logger().warn(
                    'GPS received, but waiting for /ei/odom...',
                    throttle_duration_sec=2.0
                )
            return

        if not self.origin_set:

            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.origin_alt = (
                0.0 if math.isnan(msg.altitude) else msg.altitude
            )

            self.origin_odom_x = self.latest_odom.pose.pose.position.x
            self.origin_odom_y = self.latest_odom.pose.pose.position.y
            self.origin_odom_z = self.latest_odom.pose.pose.position.z

            # Bias-corrected-only track starts at the ECC origin.
            self.scaled_x = self.origin_odom_x
            self.scaled_y = self.origin_odom_y
            self._last_odom_xy = (self.origin_odom_x, self.origin_odom_y)

            # Kalman state also starts exactly at the ECC origin --
            # by definition, not an estimate -- so its initial
            # variance is small (kf_initial_variance), not zero,
            # so the very first GPS updates can still pull it if
            # needed.
            self.kf_x = self.origin_odom_x
            self.kf_y = self.origin_odom_y
            self.kf_p_x = self.kf_initial_variance
            self.kf_p_y = self.kf_initial_variance

            self.origin_set = True

            self.get_logger().info('======================================')
            self.get_logger().info(' GPS ORIGIN INITIALIZED (single fix)')
            self.get_logger().info(f'Latitude  : {self.origin_lat:.8f}')
            self.get_logger().info(f'Longitude : {self.origin_lon:.8f}')
            self.get_logger().info(f'Altitude  : {self.origin_alt:.3f} m')
            self.get_logger().info(f'Odom X    : {self.origin_odom_x:.3f} m')
            self.get_logger().info(f'Odom Y    : {self.origin_odom_y:.3f} m')
            self.get_logger().info(
                f'Yaw offset: {math.degrees(self.yaw_offset):.3f} deg'
            )
            self.get_logger().info('======================================')
            return

        # ------------------------------------------------------------
        # GPS -> local ENU -> rotate into /ei/odom orientation
        # ------------------------------------------------------------

        east, north, up = self.gps_to_enu(
            msg.latitude,
            msg.longitude,
            0.0 if math.isnan(msg.altitude) else msg.altitude
        )

        self.gps_x = east
        self.gps_y = north
        self.gps_z = up

        cos_yaw = math.cos(self.yaw_offset)
        sin_yaw = math.sin(self.yaw_offset)

        rotated_x = cos_yaw * east - sin_yaw * north
        rotated_y = sin_yaw * east + cos_yaw * north

        gps_odom_x = self.origin_odom_x + rotated_x
        gps_odom_y = self.origin_odom_y + rotated_y
        gps_odom_z = self.origin_odom_z + up


        odom_x = self.latest_odom.pose.pose.position.x
        odom_y = self.latest_odom.pose.pose.position.y

        if self._last_sync_gps_en is not None:
            d_gps = math.hypot(
                east - self._last_sync_gps_en[0],
                north - self._last_sync_gps_en[1]
            )
            d_ecc = math.hypot(
                odom_x - self._last_sync_odom_xy[0],
                odom_y - self._last_sync_odom_xy[1]
            )

            if d_gps >= self.scale_min_step_m and d_ecc >= self.scale_min_step_m:
                self._scale_sum_xy += d_ecc * d_gps
                self._scale_sum_xx += d_ecc * d_ecc
                self._scale_pair_count += 1

                if self._scale_sum_xx > 0.0:
                    self.ecc_scale_raw = self._scale_sum_xy / self._scale_sum_xx

                    if self._scale_pair_count >= self.scale_warmup_pairs:
                        a = self.scale_smoothing_alpha
                        self.ecc_scale = (
                            (1.0 - a) * self.ecc_scale + a * self.ecc_scale_raw
                        )

        self._last_sync_gps_en = (east, north)
        self._last_sync_odom_xy = (odom_x, odom_y)

        if len(msg.position_covariance) == 9 and msg.position_covariance[0] > 0.0:
            r_x = msg.position_covariance[0]
            r_y = msg.position_covariance[4]
        else:
            r_x = self.gps_measurement_noise_std ** 2
            r_y = r_x

        k_x = self.kf_p_x / (self.kf_p_x + r_x)
        k_y = self.kf_p_y / (self.kf_p_y + r_y)

        self.kf_x += k_x * (gps_odom_x - self.kf_x)
        self.kf_y += k_y * (gps_odom_y - self.kf_y)
        self.kf_p_x = (1.0 - k_x) * self.kf_p_x
        self.kf_p_y = (1.0 - k_y) * self.kf_p_y

        # ------------------------------------------------------------
        # Publish bias-corrected-only ECC track (/ei/odom_scaled)
        # ------------------------------------------------------------

        scaled_msg = Odometry()
        scaled_msg.header.stamp = self.latest_odom.header.stamp
        scaled_msg.header.frame_id = 'odom'
        scaled_msg.child_frame_id = 'base_footprint'
        scaled_msg.pose.pose.position.x = self.scaled_x
        scaled_msg.pose.pose.position.y = self.scaled_y
        scaled_msg.pose.pose.position.z = self.latest_odom.pose.pose.position.z
        scaled_msg.pose.pose.orientation = self.latest_odom.pose.pose.orientation
        scaled_msg.twist.twist.linear.x = self.latest_odom.twist.twist.linear.x * self.ecc_scale
        scaled_msg.twist.twist.linear.y = self.latest_odom.twist.twist.linear.y * self.ecc_scale
        scaled_msg.twist.twist.angular = self.latest_odom.twist.twist.angular

        self.ecc_scaled_pub.publish(scaled_msg)

        # ------------------------------------------------------------
        # Publish Kalman-fused GPS+ECC odometry (/gps/odom)
        # ------------------------------------------------------------

        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = self.kf_x
        odom_msg.pose.pose.position.y = self.kf_y
        odom_msg.pose.pose.position.z = gps_odom_z

        # GPS provides no orientation; keep ECC's.
        odom_msg.pose.pose.orientation = self.latest_odom.pose.pose.orientation

        # Scale-corrected velocity (GPS has none of its own here).
        odom_msg.twist.twist = scaled_msg.twist.twist

        # Position covariance now reflects the actual Kalman estimate
        # uncertainty, rather than a fixed guess.
        odom_msg.pose.covariance[0] = self.kf_p_x
        odom_msg.pose.covariance[4] = self.kf_p_y
        odom_msg.pose.covariance[8] = 25.0  # Z still unmodeled by this filter

        odom_msg.pose.covariance[35] = 99999.0  # GPS does NOT determine orientation

        odom_msg.twist.covariance[0] = 99999.0
        odom_msg.twist.covariance[7] = 99999.0
        odom_msg.twist.covariance[14] = 99999.0
        odom_msg.twist.covariance[21] = 99999.0
        odom_msg.twist.covariance[28] = 99999.0
        odom_msg.twist.covariance[35] = 99999.0

        self.gps_odom_pub.publish(odom_msg)

        self.debug_print(msg, east, north, up, gps_odom_x, gps_odom_y, gps_odom_z, k_x, k_y)

    # ================================================================
    # GPS -> ENU
    # ================================================================

    def gps_to_enu(self, lat, lon, alt):

        lat0 = math.radians(self.origin_lat)
        lon0 = math.radians(self.origin_lon)

        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        d_lat = lat_rad - lat0
        d_lon = lon_rad - lon0

        east = d_lon * math.cos(lat0) * self.EARTH_RADIUS
        north = d_lat * self.EARTH_RADIUS
        up = alt - self.origin_alt

        return east, north, up

    # ================================================================
    # DEBUG OUTPUT
    # ================================================================

    def debug_print(self, gps_msg, east, north, up,
                     gps_odom_x, gps_odom_y, gps_odom_z, k_x, k_y):

        if not self.debug:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_debug_time).nanoseconds / 1e9
        if elapsed < self.debug_period:
            return
        self.last_debug_time = now

        odom_x = self.latest_odom.pose.pose.position.x
        odom_y = self.latest_odom.pose.pose.position.y
        odom_z = self.latest_odom.pose.pose.position.z

        dx = self.kf_x - odom_x
        dy = self.kf_y - odom_y
        distance = math.hypot(dx, dy)

        self.get_logger().info(
            '\n'
            '------------- GPS ODOM DEBUG -------------\n'
            f'GPS lat/lon : {gps_msg.latitude:.8f}, {gps_msg.longitude:.8f}\n'
            f'GPS ENU     : E={east:+.3f}  N={north:+.3f}  U={up:+.3f}\n'
            f'GPS raw pos : X={gps_odom_x:+.3f}  Y={gps_odom_y:+.3f}\n'
            f'ECC scale   : smoothed={self.ecc_scale:.4f} '
            f'({(self.ecc_scale - 1.0) * 100:+.2f} %)  '
            f'raw={self.ecc_scale_raw:.4f}  pairs={self._scale_pair_count}\n'
            f'KF gain     : Kx={k_x:.3f}  Ky={k_y:.3f}\n'
            f'KF variance : Px={self.kf_p_x:.4f}  Py={self.kf_p_y:.4f}\n'
            f'KF fused    : X={self.kf_x:+.3f}  Y={self.kf_y:+.3f}  Z={gps_odom_z:+.3f}\n'
            f'/ei/odom    : X={odom_x:+.3f}  Y={odom_y:+.3f}  Z={odom_z:+.3f}\n'
            f'KF - ECC    : dX={dx:+.3f}  dY={dy:+.3f}  dist={distance:.3f} m\n'
            '-------------------------------------------'
        )


def main(args=None):

    rclpy.init(args=args)
    node = GPSOdom()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()