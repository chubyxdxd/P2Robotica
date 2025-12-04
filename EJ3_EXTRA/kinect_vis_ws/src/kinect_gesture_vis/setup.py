from setuptools import find_packages, setup

package_name = 'kinect_gesture_vis'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='miguelsilva',
    maintainer_email='miguel.silva@ucb.edu.bo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gesture_detector = kinect_gesture_vis.gesture_detector_node:main',
            'gesture_heat_map = kinect_gesture_vis.gesture_heatmap_node:main',

        ],
    },
)
