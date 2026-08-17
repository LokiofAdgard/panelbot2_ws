from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_share = get_package_share_directory('odom_estimator')
    navsat_config = os.path.join(pkg_share, 'config', 'navsat.yaml')

    navsat = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[navsat_config],
        remappings=[
            ('gps/fix', '/fix'),
            ('gps/filtered', '/gps/filtered'),
            ('odometry/filtered', '/ecc/odom')
        ]
    )

    return LaunchDescription([navsat])
