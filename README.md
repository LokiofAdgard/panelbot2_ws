# panelbot2_ws

ROS 2 workspace for a solar-panel robot that localises itself in feature-poor environments. The robot's starting pose comes from an AprilTag, and after that it tracks its motion with visual odometry (ORB or ECC) from a downward-facing camera, fused with IMU heading.

Tested on a Raspberry Pi 4B running Ubuntu 24.04 and ROS 2 Jazzy, with an ESP32-S3 main controller running micro-ROS.

## Packages

| Package | Purpose |
|---|---|
| `mros_interfaces` | Custom message definitions for the data published by the ESP32 over micro-ROS (e.g. `Imu`) |
| `mros_converter` | Launches the micro-ROS agent and the bridge that converts ESP32 sensor messages to ROS 2 types |
| `odom_estimator` | Visual odometry node and AprilTag initial pose node |

## Build

```bash
cd panelbot2_ws
colcon build
source install/setup.bash
```

Dependencies: `rclpy`, `cv_bridge`, `tf2_ros`, OpenCV (with `cv2.aruco`), NumPy, and the micro-ROS agent. `mros_interfaces` is built as part of the workspace.

## Usage

Run each of the following in its own terminal, after sourcing the workspace. A camera driver publishing `/image_raw` must also be running.

### 1. micro-ROS agent and bridge

```bash
ros2 launch mros_converter launch_converter.launch.py
```

This starts the micro-ROS agent over serial to connect to the ESP32, plus the bridge node that republishes the sensor data (IMU, ToF, etc.) in the formats the odometry nodes need. If the ESP32 enumerates on a different port, edit the serial device in the launch file.

### 2. Initial pose (AprilTag)

```bash
ros2 run odom_estimator initial_pos
```

This node detects an AprilTag (family 36h11, ID 0, 48 mm side) in `/image_raw`, estimates the camera pose relative to it, and publishes a **static `map → odom` transform** once. After that it ignores further images.

The tag ID, tag size, camera intrinsics, axis mapping, and offsets are constants at the top of `initial_pos.py`. Edit them there if your setup differs.

### 3. Visual odometry

```bash
ros2 run odom_estimator ei_tracker --ros-args \
  -p motion_estimator:=orb \
  -p process_mode:=default
```

This node estimates frame-to-frame motion from `/image_raw`, takes heading from the IMU, and publishes the `odom → base_footprint` transform along with an odometry message.

## Parameters (`ei_tracker`)

| Parameter | Default | Description |
|---|---|---|
| `motion_estimator` | `orb` | Visual estimator to use: `orb` or `ecc` |
| `process_mode` | `default` | Image preprocessor: `default`, `glare_suppress`, or `periodicity_dampen` |
| `direction` | `reverse_x` | Camera-to-robot axis mapping: `forward_x`, `reverse_x`, `forward_y`, or `reverse_y` |
| `yaw_weight_imu` | `1.0` | Weight of IMU yaw in the fused heading |
| `yaw_weight_ecc` | `0.0` | Weight of visual yaw in the fused heading |
| `imu_yaw_smoothing` | `0.1` | Exponential smoothing gain for IMU yaw |
| `ecc_score_min` | `0.5` | Frames with a match score below this are skipped (applies to both estimators) |
| `motion_deadband_px` | `0.15` | Pixel motion below this is treated as zero |
| `orb_n_features` | `500` | Maximum ORB keypoints per frame |
| `orb_match_ratio` | `0.8` | Lowe's ratio test threshold |
| `orb_ransac_thresh_px` | `4.0` | RANSAC reprojection threshold (px) |
| `orb_min_inliers` | `8` | Minimum inliers to accept a frame |
| `orb_fast_threshold` | `7` | FAST detector threshold |
| `orb_edge_threshold` | `15` | ORB edge border size |

The pixel-to-metre scale factor (≈ 6138.8 px/m) is set in `publish()` in `ei_tracker.py`. It was calibrated at a camera height of 76 mm, so recalibrate it if the camera height changes.

## Topics and transforms

| Node | Subscribes | Publishes |
|---|---|---|
| `initial_pos` | `/image_raw` | TF (static): `map → odom` |
| `ei_tracker` | `/image_raw`, `/imu/data` | `/ei/odom`, `/ei/debug_image`, TF: `odom → base_footprint` |

`/ei/debug_image` shows the ORB inlier features overlaid on the camera image, which you can view in RViz.

## Related repositories

- [PBot Main Controller](https://github.com/LokiofAdgard/PBot_Main_Controller) (ESP32-S3, micro-ROS)
- [PBot Motor Controller](https://github.com/LokiofAdgard/PBot_Motor_Controller) (STM32F103)
- [PBot Power Controller](https://github.com/LokiofAdgard/PBot_Power_Controller) (STM32F103)