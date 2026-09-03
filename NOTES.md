# Notes: asu_il_policy_ws

Build rationale, known gotchas ("don't re-debug these"), usage commands,
and next-phase design. See [README.md](README.md) for project purpose and
status.

## Why this image, not a simpler one

- Base image: `hri5792:5000/v03:24.04.cudagl-core-py-noetic.src-jazzy.brg`
  (Ubuntu 24.04 + CUDA/GL + Python 3.12.3 + ROS1 Noetic src + ROS2 Jazzy +
  ros1_bridge) -- same base as `asu_state_reward_ws`. We only need ROS2
  (rclpy); noetic/bridge just come along for free from the shared base.
- Rejected alternative: `hri5792:5000/v03.cv50:...-jazzy-torch-vlm` (pure
  jazzy, torch==2.13.0+cu130 preinstalled). Tempting but rejected to avoid an
  unverified newer torch + unknown "vlm" extras; instead we pin the exact
  torch/torchvision build that's already verified working (the host's `aloha`
  conda env, see below).
- **No conda inside the container.** ROS2's `rclpy` is tied to the system
  (apt) Python; porting conda in would give two interpreters that can't share
  `rclpy` + `torch` in one process. Instead we `pip install` (into the
  container's system Python, `/usr/local/lib/python3.12/dist-packages`) the
  exact package versions verified in the host's `aloha` conda env
  (`/home/asu/miniconda3/envs/aloha`, built for
  `git@github.com:sszxc/act.git`). Pinned list:
  `docker/build/requirements.txt`. torch/torchvision are installed separately
  in `user.Dockerfile` from the cu128 index -- the RTX PRO 6000 Blackwell GPU
  needs a cu128+ build to be recognized at all.

## Bugs found & fixed while building the image (don't re-debug these)

1. **`pip install -r requirements.txt` fails with `uninstall-no-record-file`.**
   The base image ships several packages (numpy, PyOpenGL, scipy, matplotlib,
   pillow, sympy, mpmath, ...) as dpkg-managed system packages with no pip
   RECORD file, so pip's default upgrade-install (uninstall-then-install)
   errors out. Fix: `--ignore-installed` on both pip installs in
   `user.Dockerfile`.
2. **`cv2.__version__` silently wrong (`4.14.0` instead of `5.0.0`) despite
   `pip show opencv-python` correctly showing `5.0.0.93` installed.** The base
   image's apt package `libopencv-dev` drops its own Python bindings straight
   into `dist-packages` (`dpkg -S` confirms it owns `cv2/__init__.py` there).
   `pip install --ignore-installed` overwrites the files the new wheel ships,
   but leaves apt-only leftovers behind -- specifically `cv2/config-3.12.py`,
   which patches the native-lib search path and made the old apt-installed
   OpenCV core load instead. Fix: `rm -rf .../dist-packages/cv2` immediately
   before installing requirements.txt (apt's C/C++ libs/headers elsewhere are
   untouched -- only the Python package dir is wiped). A build-time assertion
   (`assert cv2.__version__ == '5.0.0'`) is baked into `user.Dockerfile` so
   any regression fails the build loudly instead of shipping silently broken.
3. Minor/accepted, not fixed: pip resolver warns `deprecated 1.2.14 requires
   wrapt<2, but you have wrapt 2.4.0`. ACT's training code doesn't use the
   `deprecated` package -- harmless, left as-is.

## Mounts / directory layout

`start_docker.sh` -> `docker/run/run_docker.sh` bind-mounts this whole repo
as the container's `$HOME` (`/home/asu`), plus two project-specific mounts:

- **`/home/asu/code/act` -> `$HOME/act`** (rw, live bind mount, added to
  `run_docker.sh`). The `act` repo (`git@github.com:sszxc/act.git`) is
  **not** cloned into this workspace -- it's 78G on the host, of which 76G is
  `data/`, and `data/` (and `results/`) are gitignored/untracked, so a
  `git clone` would only get the code and still require mounting the data
  separately anyway. `imitate_episodes.py` also resolves `data/...` and
  `ckpt_dir=results/...` as paths relative to cwd (`hydra.run.dir: .`), so
  code+data+training-output need to live together as one tree either way.
  Net effect: zero-copy, host `git` operations work normally on the one true
  copy, checkpoints written during training land directly on host disk and
  survive the container being removed.
- **`/home/asu/code/Honda_proto5_description` -> `$HOME/Honda_proto5_description`**
  (rw, live bind mount, added to `run_docker.sh` for `act_infer_mujoco`). Same
  reasoning as the `act` mount: lives outside this workspace repo, so it's
  bind-mounted rather than copied in. **The currently-running container was
  started before this mount was added -- restart it
  (`./start_docker.sh -d --name asu_il_policy`) to pick it up** before running
  `act_infer_mujoco` with the viewer enabled.
- `detr` (`act/detr/`) has no PyPI release and isn't baked into the image
  (it lives in the bind-mounted, not image-baked, `act/` dir). It's
  `pip install -e $HOME/act/detr --no-deps` (installs to `~/.local`, no root
  needed) on **every** container start -- see the user-editable block in
  `docker/run/entrypoint.sh`. Cheap (detr declares no deps of its own).
- `project_data_container` / `/data` in `project.conf` is the template's
  generic large-data mount point; **unused** here since `act/data` already
  covers it via the mount above.

## Training

Build/run:
```bash
./docker/build/build.sh     # --no-cache, tags v03:..._asu_il_policy_$USER
./start_docker.sh -d --name asu_il_policy   # detached, persists (tail -f)
./exec-docker.sh -c asu_il_policy           # attach a shell
```

Inside the container, cwd matters -- run from `~/act`:
```bash
cd ~/act
python3 imitate_episodes.py task_name=real_pick_yellow_bottle \
    --ckpt_dir=results/real_pick_yellow_bottle_v2 batch_size=8 num_epochs=8000
```
Note the mixed CLI style: `task_name=`/`batch_size=`/`num_epochs=` are Hydra
overrides (`@hydra.main(config_path='config', config_name='config')`);
`--ckpt_dir`/`--eval`/`--temporal_agg`/etc. are plain argparse
(`parser.parse_known_args()` merged into the Hydra cfg in `main()`) -- both
styles are required, this isn't a typo in the example command.

Verified working end-to-end on 2026-09-02: `torch 2.11.0+cu128`,
`torch.cuda.is_available()==True`, `NVIDIA RTX PRO 6000 Blackwell Workstation
Edition` correctly detected, task config for `real_pick_yellow_bottle` found
in `aloha_scripts/constants.py` (`dataset_dir=data/real_pick_yellow_bottle/
good_41`, 41 episodes, `state_dim=action_dim=24`), loss decreasing normally
(epoch 0 val loss 78.6 -> epoch 1620 val loss 0.52), ~1.6 it/s on this GPU
(~1.5-2h for the full 8000 epochs at `batch_size=8`).

## Live data-availability monitor

`ros2_ws/src/policy_runner` (ament_python package) subscribes camera images
+ `sensor_msgs/JointState` over ROS2 and serves a browser page
(`web_monitor`, `http://<host>:8080`) showing which streams are live --
per-camera preview + online/offline, and per joint stream online/Hz/joint
count. This is the subscriber half of `act_infer_mujoco` below, built and
verified first so the topic/QoS wiring is known-good before inference is
added. UI (`static/index.html` + `app.js` + `styles.css`) and the HTTP
server in `web_monitor.py` are adapted from `trajectory_recorder`'s
(`/home/asu/Downloads/trajectory_recorder_new/`) capture console, with the
episode-recording pipeline (Start/Finish/label, HDF5/MP4) stripped out --
this package only checks that data is flowing, it never writes to disk. It
replaces an earlier rerun.io-based `viz_node`, dropped because it hadn't
been fully verified.

- `config/topics.yaml`: per-camera `topic` + `group` (which panel it
  renders in -- `external`/`wrist`/`right`/`tactile`; a panel with nothing
  assigned just renders empty), plus `joint_states` topics, transport
  (raw/compressed), and joint QoS. **Topic names are verified live**
  against univ.perception's actual "avatar" rig (right UR arm + dexterous
  hand): `/joint_states` (24 DOF, matches `state_dim=action_dim=24` for
  `real_pick_yellow_bottle`) and 4 live cameras (`camera101`, `camera102`,
  `side_cam`, `wrist_cam` -- a 5th, `camera100`, exists but wasn't
  publishing). None of these carry a `left`/`top` label anywhere in ROS
  (topic, node, or `camera_info` frame_id all just say `camera100`/`101`/
  `102`); **confirmed with user (2026-09-03): `camera100`=left,
  `camera101`=right (unused by this policy), `camera102`=top** -- encoded in
  `act_inference.camera_topic_map` in the yaml, used by `act_infer_mujoco`.
- **Gotcha found while verifying this: a stale `ros2 daemon`.** DDS
  env vars (`ROS_DOMAIN_ID`/`FASTRTPS_DEFAULT_PROFILES_FILE`) only take
  effect for a *new* daemon process -- if `ros2 <anything>` already
  auto-started one earlier in the container's life (e.g. before `.bashrc`
  picked up these vars), it silently keeps running with the old
  environment and never discovers anything, and `ros2 topic list` just
  returns the 3 always-there topics with no error. Fix: `ros2 daemon
  stop` (it auto-restarts with current env on the next `ros2` command).
  Check with `ros2 daemon status` first if discovery looks empty.
- Confirmed end-to-end against the real rig: 4 cameras decode correctly
  (`rgb8`, 480x640/480x848), `/joint_states` delivers real 24-DOF
  positions, no QoS mismatches (cameras are BEST_EFFORT, `/joint_states`
  is RELIABLE -- both match this package's defaults) or decode errors over
  a clean run/shutdown cycle.
- QoS gotcha (silent failure, no error/log if wrong): images always use
  ROS2's predefined "sensor data" profile (best-effort); `joint_states`
  defaults to `reliable` in the yaml -- if you point it at a *different*
  joint topic, check that publisher's QoS first (`ros2 topic info <topic>
  --verbose`) since a mismatch produces no error, just silence.
- **Bug found and fixed (2026-09-03): near-zero camera frame rate.** All 5
  camera subscriptions shared the node's single default
  `MutuallyExclusiveCallbackGroup` -- `MultiThreadedExecutor` still
  serializes callbacks within one group, so whichever camera was mid
  decode/JPEG-encode blocked the other 4, and DDS silently drops
  best-effort frames that arrive while the subscriber is busy. Fixed by
  giving every camera/joint subscription its own
  `MutuallyExclusiveCallbackGroup` (the pattern `trajectory_recorder`'s
  `node.py` already used, ported over now). Separately, `image_transport:
  raw` for 5 simultaneous cameras is itself unreliable on this rig's
  network -- `ros2 topic hz` on a single raw topic intermittently returned
  *zero* messages even with the monitor stopped -- confirming the
  `trajectory_recorder` README's own finding that raw multi-camera streams
  exceed available DDS/network throughput. `image_transport` now defaults
  to `compressed`; verified all 5 cameras `online` with 15-30Hz reported
  rate and <350ms staleness, sustained over repeated checks.

Build and run (inside the container):
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select policy_runner
source install/setup.bash
ros2 run policy_runner web_monitor   # serves http://<this-machine-LAN-IP>:8080
                                      # (container runs --network=host, so no
                                      # port mapping is needed)
```
`--config <path>` overrides which `topics.yaml` is loaded (default: the
installed `config/topics.yaml` in this package's share dir). `--raw` /
`--compressed` override `image_transport`.

## ACT inference -> MuJoCo visualization (`act_infer_mujoco`)

`ros2_ws/src/policy_runner`'s `act_infer_mujoco` node subscribes the live
`left`/`top` cameras + `/joint_states`, runs ACT inference at a fixed 30Hz,
and writes the policy's raw joint-target output straight into a MuJoCo
model's `qpos` for visualization. **No actuators, no `mj_step` (no physics
stepping at all -- forward kinematics only via `mj_forward`), no UDP output
to the real robot yet.** This is a look-before-you-command dev step ahead of
the still-unbuilt UDP-to-real-robot stage.

- **Files:** `policy_runner/act_model.py` (`ACTChunkPolicy` -- loads the
  checkpoint the same way `imitate_episodes.py`'s `eval_bc()` does: task
  config from `aloha_scripts/constants.py` via the checkpoint's own
  `config_hydra_resolved.yaml`, weights from the `.ckpt`, normalization from
  `dataset_stats.pkl`; two chunking modes selected by `temporal_agg`: off
  (default) buffers one action chunk at a time, re-querying the model every
  `chunk_size` steps; on queries every step and blends overlapping chunk
  predictions with an exponentially decaying weight -- same as `eval_bc()`'s
  `--temporal_agg` path, chunk_size-x the inference cost), `policy_runner/mujoco_qpos_viz.py` (`MujocoQposViz` -- loads the
  MJCF, maps a 24-dim action straight onto `qpos` by joint name, calls
  `mj_forward`, drives a `mujoco.viewer.launch_passive` window),
  `policy_runner/act_infer_mujoco.py` (the ROS2 node + entry point: same
  per-subscription `MutuallyExclusiveCallbackGroup`/`MultiThreadedExecutor`
  pattern as `web_monitor`, spun in a background thread; the main thread runs
  the fixed-30Hz inference/viewer loop since MuJoCo's passive viewer wants
  one consistent thread calling `sync()`).
- **Config:** `act_inference:` block in `config/topics.yaml` --
  `camera_names` (order matters, must match the trained task's
  `camera_names`), `camera_topic_map` (logical name -> key under `cameras:`),
  `control_hz`, `temporal_agg` (bool, default off), `temporal_agg_k` (decay
  rate, only used when `temporal_agg: true`), `ckpt_path`, `act_repo_root`,
  `mjcf_path` (all three as seen **inside the container** -- see "Mounts"
  above).
- **Joint order / qpos mapping (unverified against the live rig):** the
  given MJCF's arm+hand actuator include order (6 UR joints + `WRZ`/`WRY` +
  4 fingers x 4 joints = 24) produces a plain `qpos[0..23]` layout with no
  other joints in the scene (checked with `mujoco.MjModel` directly). Action
  index `i` is written straight to `qpos[i]` in that order. This is assumed
  to be the same order `/joint_states` publishes in (and thus the same order
  training's `qpos`/`action` vectors are in) because this MJCF is this
  robot's own joint list -- **not cross-checked against a live
  `/joint_states` message**. If the rendered pose looks wrong (e.g. fingers
  driving the wrist), check this mapping first.
- **Verified so far:** `ACTChunkPolicy` loading + inference smoke-tested
  inside the running container against the real checkpoint on GPU -- loads
  in ~1.5s, chunk query ~0.28s (well under the ~1.67s chunk_size=50 @ 30Hz
  budget), buffered steps are ~instant, output is a sane 24-dim float
  vector. `MujocoQposViz` smoke-tested against the real MJCF on the host
  (`mujoco==3.12.0`) with a dummy action -- `mj_forward` succeeds, no
  scene/joint-name mismatches. **Not yet run:** the live-ROS + MuJoCo-viewer
  path end-to-end against the real rig (needs a container restart for the
  new `Honda_proto5_description` mount, plus an X11-capable `DISPLAY` for
  the viewer window).
- Run (inside the container, after rebuilding this package):
  ```bash
  cd ~/ros2_ws
  colcon build --symlink-install --packages-select policy_runner
  source install/setup.bash
  ros2 run policy_runner act_infer_mujoco
  ```
  `--config <path>` overrides `topics.yaml`; `--ckpt <path>` overrides
  `ckpt_path`; `--hz <n>` overrides `control_hz`; `--no-viewer` runs the
  inference loop without opening the MuJoCo window (e.g. for a headless
  smoke test).

## Next phase: UDP output to the real robot (not built yet)

Blocked on validating the MuJoCo visualization above against the live rig
first. Design agreed in chat but not yet implemented:

- **Confirmed with user:** action space = standard joint target positions
  (matches training dataset). Control frequency = **30Hz** -- NOT the stale
  `DT = 0.02` (50Hz) default in `act/constants.py`.
- **Still needed:** UDP target IP/port + packet format expected by the
  downstream machine.
- **ROS2 DDS/discovery config added** (`.bashrc` + `fastdds_profile.xml`,
  copied from `asu_state_reward_ws`'s pattern: `ROS_DOMAIN_ID=10`,
  `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, `FASTRTPS_DEFAULT_PROFILES_FILE`
  pinning the transport NIC + explicit UDP peer IPs -- the switch's IGMP
  snooping prunes normal multicast discovery), verified live against
  univ.perception (see "Live data-availability monitor" above). Known lab
  machines: `univ.perception 192.168.1.25` (cameras, likely where
  `trajectory_recorder` runs), `univ.brain.left 192.168.1.26` (this host),
  `univ.brain.right 192.168.1.27`, `univ.controller 192.168.1.12`.
- `pyrealsense2` is preinstalled in the base image -- strong hint the camera
  hardware is Intel RealSense (not yet confirmed against `topics.yaml`).

See `docker/build/requirements.txt` for the full pinned training dep list.
