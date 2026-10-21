from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Can run on a different machine than calibration_node (e.g. a laptop connected to the
    # robot). Activate the package venv before launching (see README.md).
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
