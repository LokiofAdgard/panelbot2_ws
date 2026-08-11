from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():

    micro_ros_agent = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent_serial',
        output='screen',
        arguments=['serial', '--dev', '/dev/ttyACM0', '-b', '1000000']
    )

    delayed_converter = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='mros_converter',
                executable='converter_node',
                name='converter_node',
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        micro_ros_agent,
        delayed_converter
    ])
