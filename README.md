# asu_il_policy_ws

Dockerized workspace (IRR "Dockerized Workspace" template, same pattern as
`asu_state_reward_ws`) for training an ACT (Action Chunking Transformer)
imitation-learning policy and, next, running it live: subscribe camera/joint
topics over ROS2, run inference, send action commands via UDP to another
machine on the LAN.

## Status

- [x] Docker image builds and runs training on the GPU (see "Training" below).
- [x] ROS2 DDS discovery configured (`.bashrc` + `fastdds_profile.xml`,
      copied from `asu_state_reward_ws` -- same host/network) and verified
      live against univ.perception's real rig. `ros2_ws/src/policy_runner`
      subscribes image+joint_states and serves a browser page
      (`web_monitor`) showing which streams are live, confirmed working
      end-to-end with real camera/joint data. `config/topics.yaml`'s
      cameras still need eyeballing in the monitor page to know which two
      are the trained "left"/"top" pair -- see "Live data-availability
      monitor" below.
- [ ] ROS2 `policy_runner` inference node (run ACT, send actions over UDP)
      -- design agreed, not built yet. See "Next phase" below.

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
as the container's `$HOME` (`/home/asu`), plus one project-specific mount:

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
- `detr` (`act/detr/`) has no PyPI release and isn't baked into the image
  (it lives in the bind-mounted, not image-baked, `act/` dir). It's
  `pip install -e $HOME/act/detr --no-deps` (installs to `~/.local`, no root
  needed) on **every** container start -- see the user-editable block in
  `docker/run/entrypoint.sh`. Cheap (detr declares no deps of its own).
- `project_data_container` / `/data` in `project.conf` is the template's
  generic large-data mount point; **unused** here since `act/data` already
  covers it via the mount above.
- `.bashrc` currently sources only `/opt/ros/jazzy/setup.bash` -- **no**
  `ROS_DOMAIN_ID` / Fast DDS discovery config yet (see "Next phase").

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
count. This is the subscriber half of the planned ACT inference node below,
built and verified first so the topic/QoS wiring is known-good before
inference is added. UI (`static/index.html` + `app.js` + `styles.css`) and
the HTTP server in `web_monitor.py` are adapted from
`trajectory_recorder`'s (`/home/asu/Downloads/trajectory_recorder_new/`)
capture console, with the episode-recording pipeline (Start/Finish/label,
HDF5/MP4) stripped out -- this package only checks that data is flowing,
it never writes to disk. It replaces an earlier rerun.io-based `viz_node`,
dropped because it hadn't been fully verified.

- `config/topics.yaml`: per-camera `topic` + `group` (which panel it
  renders in -- `external`/`wrist`/`right`/`tactile`; a panel with nothing
  assigned just renders empty), plus `joint_states` topics, transport
  (raw/compressed), and joint QoS. **Topic names are verified live**
  against univ.perception's actual "avatar" rig (right UR arm + dexterous
  hand): `/joint_states` (24 DOF, matches `state_dim=action_dim=24` for
  `real_pick_yellow_bottle`) and 4 live cameras (`camera101`, `camera102`,
  `side_cam`, `wrist_cam` -- a 5th, `camera100`, exists but wasn't
  publishing). **None of these carry a `left`/`top` label anywhere in ROS**
  (topic, node, or `camera_info` frame_id all just say `camera100`/`101`/
  `102`) -- there's no way to tell from ROS metadata which two are the
  `aloha_scripts/constants.py` "left"/"top" pair the model trained on.
  Run `web_monitor`, look at the live preview, and rename the two you want
  in the yaml.
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

## Next phase: ROS2 `policy_runner` inference node (not built yet)

Scope: subscribe camera image(s) + joint state over ROS2 (rclpy), run ACT
inference, send the resulting action chunk over UDP to another LAN machine.
Design agreed in chat but not yet implemented:

- **Package shape:** a lightweight ament Python package (`package.xml`,
  `setup.py`, `--symlink-install`), not folded into an existing package.
  Planned files: `config/policy.yaml`, `policy_runner/node.py` (subscribes
  image + joint_states, timer-driven control loop, staleness watchdog),
  `policy_runner/model.py` (ACT load/inference/temporal-aggregation
  wrapper), `policy_runner/udp_sender.py`.
- **Reference code for the subscriber side:**
  `/home/asu/Downloads/trajectory_recorder_new/` on this host -- verified
  working in `asu_state_reward_ws` on another lab machine, and already the
  basis for `web_monitor` above. `node.py` there
  is the `rclpy` subscriber pattern (Image/CompressedImage/JointState/
  WrenchStamped with specific QoS profiles, topic names from
  `config/topics.yaml` via `get_package_share_directory`). That
  `topics.yaml` wasn't found on this host -- it likely lives on
  `univ.perception` (see network notes below), which is probably where the
  cameras/`trajectory_recorder` actually run.
- **Policy inference contract** (`act/policy.py`,
  `ACTPolicy.__call__(self, qpos, image, actions=None, is_pad=None)`):
  needs **both** proprioception (qpos/joint state) and image, not image
  alone. Outputs an action chunk of `chunk_size` steps.
  `query_frequency = chunk_size` normally, or `1` if `temporal_agg`
  (temporal ensembling) is enabled -- governs how often the control loop
  needs to re-run inference vs. reuse buffered chunk steps.
- **Confirmed with user:** action space = standard joint target positions
  (matches training dataset). Control frequency = **30Hz** -- NOT the stale
  `DT = 0.02` (50Hz) default in `act/constants.py`.
- **Confirmed with user: camera images = `CompressedImage` transport for
  this node too, not raw** -- same as `web_monitor` (see its "Live
  data-availability monitor" debug notes above: raw drops frames with
  multiple simultaneous cameras on this network). Decode with
  `image_codec.decode_compressed_color()` (already written, RGB output for
  the model); do not add a raw-`Image` code path for this node.
- **Still needed before building this:** `checkpoint_path` +
  `dataset_stats.pkl` path (from a finished training run, e.g.
  `results/real_pick_yellow_bottle_v2/`), real `camera_names`/topic names
  (currently `["left", "top"]` per the task config, need actual ROS topic
  names), `chunk_size`, whether to enable `temporal_agg`, and the UDP target
  IP/port + packet format expected by the downstream machine.
- **ROS2 DDS/discovery config added** (`.bashrc` + `fastdds_profile.xml`,
  copied from `asu_state_reward_ws`'s pattern: `ROS_DOMAIN_ID=10`,
  `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, `FASTRTPS_DEFAULT_PROFILES_FILE`
  pinning the transport NIC + explicit UDP peer IPs -- the switch's IGMP
  snooping prunes normal multicast discovery). **Not yet exercised against
  a live publisher** -- univ.perception wasn't reachable/checked when this
  was added; confirm with `ros2 topic list` before trusting it. Known lab
  machines: `univ.perception 192.168.1.25` (cameras, likely where
  `trajectory_recorder` runs), `univ.brain.left 192.168.1.26` (this host),
  `univ.brain.right 192.168.1.27`, `univ.controller 192.168.1.12`.
- `pyrealsense2` is preinstalled in the base image -- strong hint the camera
  hardware is Intel RealSense (not yet confirmed against `topics.yaml`).

See `docker/build/requirements.txt` for the full pinned training dep list.
