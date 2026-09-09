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
cd ~/ros2_ws && colcon build --symlink-install --packages-select policy_runner && source install/setup.bash
ros2 run policy_runner act_infer_mujoco
```

## OOD 指示器（ood_monitor）

推理时可选开启，纯记录/可视化：对 qpos 和每路相机的 ACT backbone 特征分别算
k-NN 距离，并把实时点投到训练集的 UMAP 平面上。不做阈值判断、不报警、不 gate 策略。

依赖 `umap-learn`（h5py/scikit-learn 镜像里已有）。已写进
`docker/build/requirements.txt`，重建镜像后自带；当前容器里可直接装（`~/.local`
挂在 workspace 上，容器重启不丢）：
```bash
pip3 install --break-system-packages umap-learn==0.5.12
```

每个 checkpoint 建一次参考集（离线，约 41 集 × ~150 帧，几分钟）：
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run policy_runner ood_build_reference \
    --ckpt /home/asu/act/results/.../combo_hedge/policy_best.ckpt
# 默认 --data-dir 取 checkpoint 自己的 task config，输出到 <ckpt_dir>/ood_reference
# --stride 10 调采样密度；--knn-k 要和 topics.yaml 的 knn_k 一致
```

然后在 `config/topics.yaml` 里把 `ood_monitor.enabled` 设成 `true`、
`reference_dir` 指到上一步的输出目录，重新 `colcon build` 再跑
`act_infer_mujoco`，页面在 `http://<机器IP>:8081`（web_monitor 占了 8080）。

页面读法：数字是当前观测到训练集 k 个最近邻的平均距离，下面的 `x train p95`
是它跟训练集自身第 95 百分位邻距的比值 —— 约 1x 属正常，绿/黄/红分别是
<1x / <2x / ≥2x。散点图是该模态训练集的 UMAP 投影（灰）+ 实时点和最近 ~20s 轨迹。
