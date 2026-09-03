# USAGE

## Docker
```bash
./docker/build/build.sh                     # 构建镜像
./start_docker.sh -d --name asu_il_policy   # 后台启动容器
./exec-docker.sh -c asu_il_policy           # 进入容器 shell
```

## 训练 ACT
```bash
cd ~/act
python3 imitate_episodes.py task_name=real_pick_yellow_bottle \
    --ckpt_dir=results/real_pick_yellow_bottle_v2 batch_size=8 num_epochs=8000
```
`task_name=`/`batch_size=` 是 Hydra 参数，`--ckpt_dir` 是 argparse 参数，两种风格混用属正常。

## 相机/关节数据监控（web_monitor）
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select policy_runner
source install/setup.bash
ros2 run policy_runner web_monitor   # http://<机器IP>:8080
```
`--config <path>` 换配置文件；`--raw` / `--compressed` 覆盖图像传输方式（默认 compressed，raw 多路会掉帧）。

改哪些相机/关节 topic → 编辑 `ros2_ws/src/policy_runner/config/topics.yaml`，改完重新 `colcon build` 生效。

## 常见坑
- `ros2 topic list` 什么都看不到 → 大概率是 `ros2 daemon` 用了旧环境变量，跑一次 `ros2 daemon stop` 再试。
- 换了新的 joint topic 但监控页面一直显示离线 → 先 `ros2 topic info <topic> --verbose` 核对 QoS（reliable/best_effort 不匹配不会报错，只是静默收不到）。


## 推理
```bash
exit                                   # 先退出容器 shell
docker stop asu_il_policy && docker rm asu_il_policy
./start_docker.sh -d --name asu_il_policy
./exec-docker.sh -c asu_il_policy

进去之后确认新挂载生效了：
ls ~/Honda_proto5_description/mjcf/   # 应该能看到 xml 文件

然后重新跑：
cd ~/ros2_ws
colcon build --symlink-install --packages-select policy_runner
source install/setup.bash
ros2 run policy_runner act_infer_mujoco
```
