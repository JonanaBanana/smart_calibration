from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # May run on another machine than calibration_node. Activate the package venv first.
    return LaunchDescription([
        DeclareLaunchArgument(
            'calibration_node',
            default_value='/calibration_node',
            description='Calibration node to connect to (its previews, detections and '
                        'parameters).'),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/ircam_wide/decoded',
            description='Raw camera topic, used only while the raw camera view is on.'),
        Node(
            package='smart_calibration',
            executable='calibration_gui',
            name='calibration_gui',
            output='screen',
            remappings=[('image', LaunchConfiguration('image_topic'))],
            parameters=[{'calibration_node': LaunchConfiguration('calibration_node')}],
        ),
    ])
