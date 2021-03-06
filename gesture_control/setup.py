from setuptools import PackageFinder, setup

package_name = 'gesture_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[
        'gesture_control',
        'pose_estimator',
        'pose_visualizer',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.20',
        'mediapipe>=0.8.9',
        'opencv-contrib-python>=4.5.2'
    ],
    zip_safe=True,
    maintainer='Je Hon Tan',
    maintainer_email='jehontan@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimator = pose_estimator.main:main',
            'gesture_controller = gesture_control.controller:main',
            'pose_visualizer = pose_visualizer.pose_visualizer:main',
        ],
    },
)
