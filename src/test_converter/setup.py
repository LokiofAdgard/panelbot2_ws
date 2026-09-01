import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'test_converter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lokiofadgard',
    maintainer_email='dulnathvanderbona@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'imuTest = test_converter.imuTest:main',
            'rawConverter = test_converter.rawConverter:main',
            'gpsPub = test_converter.gpsPub:main',
            'rawTofTest = test_converter.rawTofTest:main',
            'msgTest = test_converter.msgTest:main',
            'rawConverter2 = test_converter.rawConverter2:main',
            'test_april = test_converter.test_april:main',
            'test_april_viewer = test_converter.test_april_viewer:main',
            'cam_gps_test = test_converter.cam_gps_test:main',
            'sq_driver = test_converter.sq_driver:main',
        ],
    },
)
