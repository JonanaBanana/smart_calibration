from glob import glob

from setuptools import find_packages, setup

package_name = 'smart_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JonanaBanana',
    maintainer_email='example@email.com',
    description=(
        'Tunable monocular camera calibration (pinhole/plumb_bob and fisheye/equidistant) '
        'for ROS 2'
    ),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'calibration_node = smart_calibration.calibration_node:main',
            'calibration_gui = smart_calibration.calibration_gui:main',
        ],
    },
)
