from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Activate the package venv before launching (see README.md).
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Image topic of the camera to calibrate.'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('smart_calibration'), 'config', 'plumb_bob_chessboard.yaml']),
            description='calibration_node parameters: board size, preprocessing, detector '
                        'flags (see config/plumb_bob_chessboard.yaml).'),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='calibration_node log level; debug logs a line per frame.'),
        Node(
            package='smart_calibration',
            executable='calibration_node',
            name='calibration_node',
            output='screen',
            remappings=[('image', LaunchConfiguration('image_topic'))],
            parameters=[LaunchConfiguration('params_file')],
            ros_arguments=['--log-level', LaunchConfiguration('log_level')],
        ),
    ])
