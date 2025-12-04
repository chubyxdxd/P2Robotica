from setuptools import find_packages, setup
import site, sys

site.addsitedir('/home/fabricio/robotica/kinet2/venv/lib/python3.10/site-packages')
sys.path.insert(0, '/home/fabricio/robotica/kinet2/venv/lib/python3.10/site-packages')

package_name = 'kinet'

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
    entry_points={
        'console_scripts': [
            'correr = kinet.detectionyerko:main',
            'p2 = kinet.detection2:main',
            'webcam_node = kinet.camera_node:main'
        ],
    },
)
