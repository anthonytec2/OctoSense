import os
from glob import glob
from setuptools import setup

package_name = 'data_collect'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'static'),
            glob('data_collect/static/*'),
        ),
    ],
    install_requires=[
        'setuptools',
        'fastapi',
        'uvicorn[standard]',
        'pyyaml',
        'pyserial',
    ],
    zip_safe=True,
    maintainer='Anthony Bisulco',
    maintainer_email='abisulco@seas.upenn.edu',
    description='OctoSense multi-sensor data-collection controller (cameras, '
                'event cameras, LiDAR, IMU, GPS/RTK, CAN) for ROS 2.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'shutdown_on_bag_complete = data_collect.shutdown_on_bag_complete:main',
            'collection_controller = data_collect.main:main',
        ],
    },
)