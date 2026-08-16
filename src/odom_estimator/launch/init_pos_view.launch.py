from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    use_test_pub = LaunchConfiguration('use_test_pub')
    use_rviz = LaunchConfiguration('use_rviz')

    odom_estimator_share = FindPackageShare('odom_estimator').find('odom_estimator')
    rviz_config = odom_estimator_share + '/config/april_tf.rviz'

    return LaunchDescription([

        # Argument: enable/disable test_april publisher
        DeclareLaunchArgument(
            'use_test_pub',
            default_value='false',
            description='Run test_april publisher'
        ),

        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Run Rviz'
        ),

        # Conditional: test_april node
        ExecuteProcess(
            condition=IfCondition(use_test_pub),
            cmd=['ros2', 'run', 'test_converter', 'test_april'],
            output='screen'
        ),

        # Always run initial_pos node
        ExecuteProcess(
            cmd=['ros2', 'run', 'odom_estimator', 'initial_pos'],
            output='screen'
        ),

        # RViz2 with april_tf.rviz
        ExecuteProcess(
            condition=IfCondition(use_rviz),
            cmd=['rviz2', '-d', rviz_config],
            output='screen'
        ),
    ])
