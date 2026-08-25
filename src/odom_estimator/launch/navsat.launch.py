from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    odom_source_arg = DeclareLaunchArgument(
        'odom_source',
        default_value='ei',
        description='Select odometry source: ecc or ei'
    )

    odom_source = LaunchConfiguration('odom_source')

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
            ('odometry/filtered', [ '/', odom_source, '/odom' ])
        ]
    )

    return LaunchDescription([
        odom_source_arg,
        navsat
    ])
