# asu_il_policy_ws

Dockerized workspace (IRR "Dockerized Workspace" template, same pattern as
`asu_state_reward_ws`) for training an ACT (Action Chunking Transformer)
imitation-learning policy and, next, running it live: subscribe camera/joint
topics over ROS2, run inference, send action commands via UDP to another
machine on the LAN.

## Status

- [x] Docker image builds and runs training on the GPU.
- [x] ROS2 DDS discovery configured and verified live against
      univ.perception's real rig. `ros2_ws/src/policy_runner`'s
      `web_monitor` confirms camera/joint streams live end-to-end;
      `config/topics.yaml`'s `left`/`top` camera mapping is resolved
      (`camera100`=left, `camera102`=top, `camera101`=right/unused).
- [x] `act_infer_mujoco` ROS2 node: runs ACT inference at a fixed 30Hz and
      writes the raw action straight into a MuJoCo model's qpos for
      visualization -- no actuators/`mj_step`.
      Model-loading + inference verified end-to-end against the real
      checkpoint on GPU; the live-ROS + MuJoCo-viewer path itself has
      **not** been run against the live rig yet (needs a container restart
      to pick up the new `Honda_proto5_description` mount).
- [x] Send the resulting action over UDP to the real robot: implemented
      (JSON per tick, see `config/topics.yaml`'s `act_inference.udp_output`),
      but the target host/port are still unconfigured (blank) and it has
      **not** been run against a real receiver yet.

- [x] OOD indicator (`ood_monitor`, optional, off by default): per-tick k-NN
      distance for qpos + each camera's ACT backbone feature against the
      training set, plus a UMAP scatter of the live point, served at
      `:8081`. Recording/visualization only -- it never gates the policy.
      Reference set is built per checkpoint with
      `ros2 run policy_runner ood_build_reference` (see USAGE.md). The
      offline builder and the monitor/web endpoints are verified against the
      real checkpoint + training data; the live-rig path has not been run yet.

**Gotcha:** `config/topics.yaml` paths (`mjcf_path`, `ckpt_path`,
`act_repo_root`) are written as seen **inside the container** -- e.g.
`mjcf_path: /home/asu/Honda_proto5_description/...` only resolves there
because `docker/run/run_docker.sh` bind-mounts the host's
`/home/asu/code/Honda_proto5_description` to that container path. Running
tooling against these paths directly on the host will fail to find the file
unless you adjust for the host-side mount source.

See [NOTES.md](NOTES.md) for build rationale, known gotchas, usage
commands, and next-phase design details.
