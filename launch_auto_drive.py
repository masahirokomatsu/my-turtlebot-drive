import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    world_path = os.path.expanduser('~/pai-basic-project/worlds/course.world')
    
    tb3_models_path = os.path.expanduser('~/airobot_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/models')
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    os.environ["GAZEBO_MODEL_PATH"] = f"{tb3_models_path}:{existing_model_path}:/usr/share/gazebo-11/models"

    # 1. Gazeboをコースで起動
    start_gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', world_path],
        output='screen'
    )
    
    # 2. ロボットのスポーン
    model_path = os.path.join(tb3_models_path, 'turtlebot3_waffle_pi', 'model.sdf')
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'waffle_pi',
            '-file', model_path,
            '-x', '0.0',
            '-y', '-3.1',
            '-z', '0.01',
            '-Y', '0.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        start_gazebo,
        spawn_robot
    ])
