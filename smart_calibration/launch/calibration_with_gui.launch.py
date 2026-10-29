import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Calibration node and GUI on one machine. Activate the package venv before launching.
    launch_dir = os.path.join(get_package_share_directory('smart_calibration'), 'launch')

    def include(launch_file, arguments):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, launch_file)),
            launch_arguments={name: LaunchConfiguration(name) for name in arguments}.items())

    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Image topic of the camera to calibrate.'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('smart_calibration'), 'config', 'example_config.yaml']),
            description='calibration_node parameters: board size, preprocessing, detector '
                        'flags (see config/example_config.yaml).'),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='calibration_node log level; debug logs a line per frame.'),
        include('calibration.launch.py', ['image_topic', 'params_file', 'log_level']),
        include('gui.launch.py', ['image_topic']),
    ])
