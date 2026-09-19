from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Run AprilTag-based initial pose estimator
    test_april = Node(
        package='test_converter',
        executable='test_april',
        name='test_april',
        output='screen'
    )

    # Run initial_pos immediately
    initial_pos = Node(
        package='odom_estimator',
        executable='initial_pos',
        name='initial_pos',
        output='screen'
    )

    # After 5 seconds → run cam_gps_test
    cam_gps_test = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='test_converter',
                executable='cam_gps_test',
                name='cam_gps_test',
                output='screen'
            )
        ]
    )

    # After 5 seconds → run tracker
    tracker = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='odom_estimator',
                executable='ei_tracker',
                name='ei_tracker',
                output='screen',
                parameters=[
                    {"process_mode": "default"}
                ]
            )
        ]
    )

    gps_tracker = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='odom_estimator',
                executable='gps_tracker',
                name='gps_tracker',
                output='screen'
            )
        ]
    )

    # Publish static TF: base_footprint_ecc → gps_link
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_base_to_gps',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'gps_link'],
        output='screen'
    )

    # Launch navsat.launch.py
    pkg_share = get_package_share_directory('odom_estimator')
    navsat_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'navsat.launch.py')
        )
    )

    # Launch RViz from share directory
    pkg_share_test = get_package_share_directory('test_converter')
    rviz_config = os.path.join(pkg_share_test, 'config', 'bag_test.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        # test_april,
        # initial_pos,
        # cam_gps_test,
        tracker,
        static_tf,
        # gps_tracker,
        # navsat_launch,
        # rviz
    ])
