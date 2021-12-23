from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.descriptions import  ParameterValue

TURTLEBOT3_MODEL = 'burger'

pkg_gazebo_ros = FindPackageShare('gazebo_ros')
pkg_gesture_control_sim = FindPackageShare('gesture_control_sim')
world = PathJoinSubstitution([pkg_gesture_control_sim, 'worlds', 'empty_world.model'])
# urdf = Command(['xacro ', PathJoinSubstitution([pkg_gesture_control_sim, 'urdf', 'robot.xacro'])])
param_robot_description = ParameterValue(Command(['xacro ', PathJoinSubstitution([pkg_gesture_control_sim, 'urdf', 'robot.xacro'])]), value_type=str)

def generate_launch_description():
    return LaunchDescription([
        # Gazebo
        IncludeLaunchDescription(
            PathJoinSubstitution([
                pkg_gazebo_ros,
                'launch',
                'gzserver.launch.py'
            ]),
            launch_arguments={
                'world': world
            }.items()
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution([
                pkg_gazebo_ros,
                'launch',
                'gzclient.launch.py'
            ]),
        ),

        # State publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'robot_description': param_robot_description
            }],
        ),

        # Spawner
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='urdf_spawner',
            arguments=[
            '-unpause',
            '-entity',
            'test_model',
            '-topic',
            '/robot_description'
            ],
        ),
    ])