from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    imu_test = Node(
        package='test_converter',
        executable='imuTest',
        name='imu_test',
        output='screen'
    )

    tof_test = Node(
        package='test_converter',
        executable='rawTofTest',
        name='tof_test',
        output='screen'
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare('mros_converter'),
        'config',
        'imu_tof_view.rviz'
    ])

    rviz_view = ExecuteProcess(
        cmd=[
            'rviz2',
            '-d',
            rviz_config
        ],
        output='screen'
    )

    delayed_rviz = TimerAction(
        period=2.0,
        actions=[rviz_view]
    )

    return LaunchDescription([
        imu_test,
        tof_test,
        delayed_rviz
    ])
