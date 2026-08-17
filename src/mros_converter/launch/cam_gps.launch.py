from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    gps_node = Node(
        package='nmea_navsat_driver',
        executable='nmea_serial_driver',
        name='gps_driver',
        output='screen',
        parameters=[
            {'port': '/dev/ttyACM0'},
            {'baud': 115200},
            {'frame_id': 'gps_link'},
            {'useRMC': True},
            {'time_ref_source': 'gps'}
        ]
    )

    cam_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='usb_cam',
        output='screen',
        parameters=[
            {'video_device': '/dev/video0'},
            {'image_size': [1280, 720]},
            {'pixel_format': 'YUYV'},
            {'camera_frame_id': 'camera_link'}
        ]
    )

    return LaunchDescription([gps_node, cam_node])
