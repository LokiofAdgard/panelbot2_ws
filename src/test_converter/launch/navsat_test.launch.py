from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # 1️⃣ Run AprilTag-based initial pose estimator
    test_april = Node(
        package='test_converter',
        executable='test_april',
        name='test_april',
        output='screen'
    )

    # 2️⃣ Run initial_pos immediately
    initial_pos = Node(
        package='odom_estimator',
        executable='initial_pos',
        name='initial_pos',
        output='screen'
    )

    # 3️⃣ After 5 seconds → run cam_gps_test
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

    # 4️⃣ After 5 seconds → run ecc_tracker
    ecc_tracker = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='odom_estimator',
                executable='ecc_tracker',
                name='ecc_tracker',
                output='screen'
            )
        ]
    )

    # 5️⃣ Publish static TF: base_footprint_ecc → gps_link
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_base_to_gps',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint_ecc', 'gps_link'],
        output='screen'
    )

    # 6️⃣ Launch navsat.launch.py
    pkg_share = get_package_share_directory('odom_estimator')
    navsat_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'navsat.launch.py')
        )
    )

    # 7️⃣ Launch RViz from share directory
    pkg_share_test = get_package_share_directory('test_converter')
    rviz_config = os.path.join(pkg_share_test, 'config', 'navsat_test.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        test_april,
        initial_pos,
        cam_gps_test,
        ecc_tracker,
        static_tf,
        navsat_launch,
        rviz
    ])
