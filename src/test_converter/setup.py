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
        ],
    },
)
