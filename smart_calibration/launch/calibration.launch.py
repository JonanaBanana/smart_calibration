from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # The node runs on whatever `python3` is first on PATH (see setup.cfg), so activate the
    # package venv before launching (see README.md).
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/ircam_wide/decoded',
            description='Image topic of the camera to calibrate.'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('smart_calibration'), 'config', 'ircam_wide.yaml']),
            description='calibration_node parameters: board size, preprocessing, detector '
                        'flags (see config/ircam_wide.yaml).'),
        Node(
            package='smart_calibration',
            executable='calibration_node',
            name='calibration_node',
            output='screen',
            remappings=[('image', LaunchConfiguration('image_topic'))],
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
