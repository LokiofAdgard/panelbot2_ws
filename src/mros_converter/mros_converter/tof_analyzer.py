import rclpy
from rclpy.node import Node

import numpy as np
import matplotlib.pyplot as plt

from mros_interfaces.msg import Tof, EdgeData


class TofViewer(Node):

    def __init__(self, enable_plot=True):
        super().__init__('tof_viewer')

        # ==========================================================
        # Settings
        # ==========================================================
        self.front_cutoff = 300
        self.left_cutoff = 300
        self.back_cutoff = 300

        self.mtof_cutoff = 300
        self.rotation_deg = 90
        self.mirror_horizontal = False
        self.min_edge_points = 3

        self.data_filter_alpha = 0.2
        self.filtered_grid = None

        self.enable_plot = enable_plot

        self.subscription = self.create_subscription(
            Tof,
            '/tof/data',
            self.tof_callback,
            10
        )

        self.edge_pub = self.create_publisher(
            EdgeData,
            '/tof/edge_data',
            10
        )

        if self.enable_plot:
            self._init_plot()

        self.get_logger().info('ToF viewer started')
        self.get_logger().info(f'Data LPF alpha = {self.data_filter_alpha}')

    def _init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()

        initial_data = np.zeros((8, 8))
        self.image = self.ax.imshow(
            initial_data,
            vmin=0.0,
            vmax=self.mtof_cutoff,
            origin='upper'
        )

        self.ax.set_xlabel('ToF Columns')
        self.ax.set_ylabel('ToF Rows')

        self.ax.text(-0.18, 1.0, 'ROBOT FRONT',
                     transform=self.ax.transAxes,
                     rotation=90, va='top', ha='center',
                     fontsize=10, fontweight='bold')

        self.ax.text(-0.18, 0.0, 'ROBOT REAR',
                     transform=self.ax.transAxes,
                     rotation=90, va='bottom', ha='center',
                     fontsize=10, fontweight='bold')

        self.fig.colorbar(self.image, ax=self.ax, label='Distance')

        self.edge_line, = self.ax.plot([], [], linewidth=2, label='Detected edge')
        self.ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show(block=False)

    # ==========================================================
    # Filtering
    # ==========================================================
    def filter_tof_data(self, grid):
        if self.filtered_grid is None:
            self.filtered_grid = grid.copy()
        else:
            alpha = self.data_filter_alpha
            self.filtered_grid = alpha * grid + (1.0 - alpha) * self.filtered_grid

        return self.filtered_grid.copy()

    # ==========================================================
    # Orientation
    # ==========================================================
    def orient_grid(self, grid):
        if self.rotation_deg in (90, 180, 270):
            grid = np.rot90(grid, k=self.rotation_deg // 90)
        elif self.rotation_deg != 0:
            self.get_logger().warning(
                f'Invalid rotation_deg={self.rotation_deg}. Using original orientation.'
            )

        if self.mirror_horizontal:
            grid = np.fliplr(grid)

        return grid

    # ==========================================================
    # Edge Detection
    # ==========================================================
    def find_edge_points(self, grid):
        valid = grid >= self.mtof_cutoff
        edge_points = []
        rows, cols = valid.shape

        # Horizontal boundaries
        for y in range(rows):
            for x in range(cols - 1):
                if valid[y, x] != valid[y, x + 1]:
                    edge_points.append((x + 0.5, y))

        # Vertical boundaries
        for y in range(rows - 1):
            for x in range(cols):
                if valid[y, x] != valid[y + 1, x]:
                    edge_points.append((x, y + 0.5))

        return edge_points

    def fit_edge_line(self, edge_points):
        if len(edge_points) < self.min_edge_points:
            return None

        points = np.array(edge_points, dtype=np.float32)
        center = np.mean(points, axis=0)
        centered = points - center

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]

        # Ensure consistent orientation
        if direction[1] > 0:
            direction = -direction

        dx, dy = direction
        angle_deg = np.degrees(np.arctan2(dx, -dy))

        return center, direction, angle_deg

    def draw_edge_line(self, center, direction):
        length = 10.0
        x0 = center[0] - direction[0] * length
        y0 = center[1] - direction[1] * length
        x1 = center[0] + direction[0] * length
        y1 = center[1] + direction[1] * length

        self.edge_line.set_data([x0, x1], [y0, y1])
        self.edge_line.set_visible(True)

    def tof_callback(self, msg):

        # Simple 3-sensor check
        front_ok = msg.tof_front > self.front_cutoff
        left_ok = msg.tof_left > self.left_cutoff
        back_ok = msg.tof_back > self.back_cutoff

        count_above = sum([front_ok, left_ok, back_ok])
        result = count_above >= 3

        self.get_logger().info(
            f'Front={msg.tof_front:.3f} [{front_ok}] | '
            f'Left={msg.tof_left:.3f} [{left_ok}] | '
            f'Back={msg.tof_back:.3f} [{back_ok}] | '
            f'Count={count_above}/3 | RESULT={result}'
        )

        # Multi-ToF grid
        data = np.array(msg.mtof_right, dtype=np.float32)
        if len(data) != 64:
            self.get_logger().warning(f'Expected 64 mtof_right values, got {len(data)}')
            return

        raw_grid = data.reshape((8, 8))
        filtered_grid = self.filter_tof_data(raw_grid)
        grid = self.orient_grid(filtered_grid)

        edge_points = self.find_edge_points(grid)
        edge_result = self.fit_edge_line(edge_points)

        avg_edge_tof = self.compute_average_edge_tof(edge_points, grid)
        
        # ==========================================================
        # Publish EdgeData message
        # ==========================================================
        edge_msg = EdgeData()

        edge_msg.front_tof = front_ok
        edge_msg.left_tof = left_ok
        edge_msg.rear_tof = back_ok

        edge_msg.right_dist = avg_edge_tof if avg_edge_tof is not None else 0.0

        if edge_result is not None:
            _, _, angle_deg = edge_result
            edge_msg.right_theta = float(angle_deg)
        else:
            edge_msg.right_theta = 0.0

        self.edge_pub.publish(edge_msg)

        # ==========================================================
        # Plot update (optional)
        # ==========================================================
        if self.enable_plot:
            self._update_plot(grid, edge_points, edge_result)

    # ==========================================================
    # Plot Update
    # ==========================================================
    def _update_plot(self, grid, edge_points, edge_result):

        if edge_result is not None:
            center, direction, angle_deg = edge_result
            self.draw_edge_line(center, direction)
            edge_text = f'Edge angle: {angle_deg:+.2f}°'
        else:
            self.edge_line.set_visible(False)
            edge_text = f'Edge angle: insufficient data ({len(edge_points)} points)'

        display_grid = grid.copy()
        display_grid[display_grid > self.mtof_cutoff] = np.nan

        self.image.set_data(display_grid)

        vmax = np.nanmax(display_grid) if np.any(~np.isnan(display_grid)) else self.mtof_cutoff + 0.01
        if vmax <= self.mtof_cutoff:
            vmax = self.mtof_cutoff + 0.01

        self.image.set_clim(vmin=self.mtof_cutoff, vmax=vmax)

        self.ax.set_title(
            f'Multi-Area ToF Right | Cutoff={self.mtof_cutoff:.0f} | '
            f'Rot={self.rotation_deg}° | Mirror={self.mirror_horizontal}\n'
            f'LPF α={self.data_filter_alpha:.2f} | {edge_text}'
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def compute_average_edge_tof(self, edge_points, grid):
        if len(edge_points) == 0:
            return None

        weighted_values = []

        rows, cols = grid.shape

        for x, y in edge_points:

            # ==========================================================
            # Vertical edge: between two columns
            # ==========================================================
            if x % 1 != 0:
                row = int(y)
                left_col = int(np.floor(x))
                right_col = left_col + 1

                if 0 <= row < rows:

                    # Left cell
                    if 0 <= left_col < cols:
                        value = grid[row, left_col]

                        if value <= self.mtof_cutoff:
                            n = left_col
                            weighted_value = value * np.sin(
                                np.deg2rad((0.5 + n) * 60.0 / 7.0)
                            )
                            weighted_values.append(weighted_value)

                    # Right cell
                    if 0 <= right_col < cols:
                        value = grid[row, right_col]

                        if value <= self.mtof_cutoff:
                            n = right_col
                            weighted_value = value * np.sin(
                                np.deg2rad((0.5 + n) * 60.0 / 7.0)
                            )
                            weighted_values.append(weighted_value)

            # ==========================================================
            # Horizontal edge: between two rows
            # ==========================================================
            elif y % 1 != 0:
                col = int(x)
                top_row = int(np.floor(y))
                bottom_row = top_row + 1

                if 0 <= col < cols:

                    # Top cell
                    if 0 <= top_row < rows:
                        value = grid[top_row, col]

                        if value <= self.mtof_cutoff:
                            n = col
                            weighted_value = value * np.sin(
                                np.deg2rad((0.5 + n) * 60.0 / 7.0)
                            )
                            weighted_values.append(weighted_value)

                    # Bottom cell
                    if 0 <= bottom_row < rows:
                        value = grid[bottom_row, col]

                        if value <= self.mtof_cutoff:
                            n = col
                            weighted_value = value * np.sin(
                                np.deg2rad((0.5 + n) * 60.0 / 7.0)
                            )
                            weighted_values.append(weighted_value)

        if len(weighted_values) == 0:
            return None

        return float(np.mean(weighted_values))


def main(args=None):
    rclpy.init(args=args)

    node = TofViewer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.enable_plot:
                plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if node.enable_plot:
            plt.close('all')


if __name__ == '__main__':
    main()
