ll
pwd
cd act/
nvidia-smi
ll
nvidia-smi
uname -a
ros2 topic list
cd ~/ros2_ws
colcon build --symlink-install --packages-select policy_runner
source install/setup.bash
ros2 run policy_runner viz_node
ros2 topic list
exit
