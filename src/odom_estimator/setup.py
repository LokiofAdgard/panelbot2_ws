from setuptools import find_packages, setup

package_name = 'odom_estimator'

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
            'initial_pos = odom_estimator.initial_pos:main',
            'odom_tracker = odom_estimator.odom_tracker:main',
        ],
    },
)
