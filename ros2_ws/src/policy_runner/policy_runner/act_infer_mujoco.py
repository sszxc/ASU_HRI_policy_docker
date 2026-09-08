"""ACT inference node: subscribes the `left`/`top` camera + joint_states topics the
`real_pick_yellow_bottle` checkpoint was trained on, runs ACT at a fixed control_hz,
and writes the policy's raw joint-target output straight into a MuJoCo model's qpos
for visualization, alongside a translucent "shadow" duplicate of the robot driven by
the real observed qpos (the policy's own input) for a real-vs-target overlay. The
same action is also broadcast over UDP as JSON for real-robot control -- no
actuators/mj_step here, that step happens on the receiving machine.

This is the inference half of README "Next phase" with the real-robot-output stage
swapped for a MuJoCo viewer (a visualize-before-you-command dev step). Camera/
joint_states subscription follows `web_monitor.py`'s pattern (per-subscription
MutuallyExclusiveCallbackGroup on a MultiThreadedExecutor, so one topic decoding
never blocks another). The ROS executor spins in a background thread; the main
thread owns the fixed-rate inference/viewer loop, since MuJoCo's passive viewer
wants one consistent thread calling `sync()`.

`--rerun` additionally opens a rerun.io viewer with one timeseries plot per
joint, overlaying real qpos (logged in `_on_joint_state`, at whatever rate
/joint_states publishes) against the target sent over UDP (logged at the
control-loop's UDP-send point) as two scatter series.
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import rerun as rr
import rerun.blueprint as rrb
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState

from policy_runner.act_model import ACTChunkPolicy
from policy_runner.image_codec import decode_compressed_color
from policy_runner.mujoco_qpos_viz import JOINT_ORDER, MujocoQposViz
from policy_runner.udp_joint_sender import UdpJointSender
from policy_runner.web_monitor import load_config

# rerun per-joint plots (--rerun): real observed qpos vs. policy's UDP-sent target,
# as two scatter series overlaid on one timeseries chart per joint. Without an
# explicit blueprint, rerun's auto-layout dumps every `joints/**` scalar into one
# combined view instead of one-plot-per-joint -- the Grid of per-joint
# TimeSeriesViews below is what actually splits them out.
_RERUN_SERIES_STYLE = {
    "real": {"color": [66, 133, 244], "marker": "Circle"},
    "target": {"color": [251, 140, 0], "marker": "Cross"},
}
_RERUN_GRID_COLUMNS = 6


def init_rerun_joint_plots():
    """Spawn the rerun viewer with a Grid blueprint (one TimeSeriesView per
    joint), then log per-joint scatter styling once (static) so the real/target
    legend and colors are set before any scalar data arrives."""
    views = [rrb.TimeSeriesView(origin=f"joints/{name}", name=name) for name in JOINT_ORDER]
    blueprint = rrb.Blueprint(rrb.Grid(*views, grid_columns=_RERUN_GRID_COLUMNS))
    rr.init("act_infer_mujoco", spawn=True)
    # rr.init(default_blueprint=...) only applies if this app_id has no active
    # blueprint yet -- a prior run without one leaves the viewer's auto-generated
    # layout "active" forever after, silently ignoring the new default. send_blueprint
    # with make_active=True forces this Grid layout on regardless of that history.
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    for name in JOINT_ORDER:
        for series, style in _RERUN_SERIES_STYLE.items():
            rr.log(
                f"joints/{name}/{series}",
                rr.SeriesPoints(colors=[style["color"]], markers=[style["marker"]], names=[series]),
                static=True,
            )


class InferInputNode(Node):
    """Subscribes just the two training cameras + joint_states and caches the latest
    of each; the timed loop in main() reads it via latest_observation()."""

    def __init__(self, config, rerun_enabled=False):
        super().__init__("act_infer_mujoco")
        act_cfg = config["act_inference"]
        self.camera_names = act_cfg["camera_names"]
        self.online_timeout_ns = int(float(config["online_timeout_seconds"]) * 1e9)
        self.rerun_enabled = rerun_enabled

        self.lock = threading.Lock()
        self.latest_image = {}
        self.latest_image_ns = {}
        self.latest_qpos = None
        self.latest_qpos_ns = None
        self._joint_idx = None  # lazy: /joint_states name-order -> JOINT_ORDER index map
        self._callback_groups = []  # keep references alive -- rclpy doesn't hold them

        for cam_name in self.camera_names:
            cam_key = act_cfg["camera_topic_map"][cam_name]
            topic = config["cameras"][cam_key]["topic"]
            topic = topic if topic.endswith("/compressed") else topic.rstrip("/") + "/compressed"
            group = MutuallyExclusiveCallbackGroup()
            self._callback_groups.append(group)
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, c=cam_name: self._on_image(c, msg),
                qos_profile_sensor_data,
                callback_group=group,
            )
            self.get_logger().info(f"camera '{cam_name}' ({cam_key}) -> {topic}")

        joint_spec = next(iter(config["joint_states"].values()))
        reliability = {
            "reliable": ReliabilityPolicy.RELIABLE,
            "best_effort": ReliabilityPolicy.BEST_EFFORT,
        }[str(joint_spec["qos"]["reliability"]).lower()]
        qos = QoSProfile(
            reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=int(joint_spec["qos"]["depth"])
        )
        group = MutuallyExclusiveCallbackGroup()
        self._callback_groups.append(group)
        self.create_subscription(JointState, joint_spec["topic"], self._on_joint_state, qos, callback_group=group)
        self.get_logger().info(f"joint_states -> {joint_spec['topic']}")

    def _on_image(self, cam_name, msg):
        try:
            image = decode_compressed_color(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"camera '{cam_name}' decode error: {exc}", throttle_duration_sec=2.0)
            return
        with self.lock:
            self.latest_image[cam_name] = image
            self.latest_image_ns[cam_name] = time.time_ns()

    def _on_joint_state(self, msg):
        idx = self._joint_index_map(msg.name)
        if idx is None:
            return
        positions = np.array(msg.position, dtype=np.float32)[idx]
        now_ns = time.time_ns()
        with self.lock:
            self.latest_qpos = positions
            self.latest_qpos_ns = now_ns
        if self.rerun_enabled:
            rr.set_time("wall_time", timestamp=now_ns / 1e9)
            for name, position in zip(JOINT_ORDER, positions):
                rr.log(f"joints/{name}/real", rr.Scalars([float(position)]))

    def _joint_index_map(self, names):
        """Lazily built once: reorders a JointState's `position` array from its own
        `name` order into JOINT_ORDER. Confirmed live (2026-09-03) that the real
        robot's /joint_states does NOT publish in JOINT_ORDER -- hand/wrist joints
        come in a different order (WRZ/WRY last, not right after the arm) -- and the
        training checkpoint's dataset_stats.pkl confirms JOINT_ORDER is what qpos/
        action were actually trained on. Positional (unreordered) reads were silently
        feeding the policy a scrambled hand/wrist state."""
        if self._joint_idx is not None:
            return self._joint_idx
        name_to_i = {n: i for i, n in enumerate(names)}
        missing = [n for n in JOINT_ORDER if n not in name_to_i]
        if missing:
            self.get_logger().error(
                f"/joint_states is missing joints needed for JOINT_ORDER: {missing}", throttle_duration_sec=2.0
            )
            return None
        self._joint_idx = np.array([name_to_i[n] for n in JOINT_ORDER])
        return self._joint_idx

    def latest_observation(self):
        """Returns (qpos, images_by_camera) if qpos + every camera is fresh, else None."""
        now_ns = time.time_ns()
        with self.lock:
            if self.latest_qpos is None or now_ns - self.latest_qpos_ns > self.online_timeout_ns:
                return None
            images = {}
            for cam in self.camera_names:
                ts = self.latest_image_ns.get(cam)
                if ts is None or now_ns - ts > self.online_timeout_ns:
                    return None
                images[cam] = self.latest_image[cam]
            return self.latest_qpos.copy(), images


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="path to topics.yaml (default: installed config/topics.yaml)")
    parser.add_argument("--ckpt", default="", help="override act_inference.ckpt_path")
    parser.add_argument("--hz", type=float, default=0.0, help="override act_inference.control_hz")
    parser.add_argument(
        "--no-viewer", action="store_true", help="run inference without opening the MuJoCo viewer window"
    )
    parser.add_argument("--udp-host", default="", help="override act_inference.udp_output.host")
    parser.add_argument("--udp-port", type=int, default=0, help="override act_inference.udp_output.port")
    parser.add_argument("--no-udp", action="store_true", help="disable UDP joint output entirely")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="log real (per joint_states) vs. target (per UDP send) qpos to a rerun.io viewer, one plot per joint",
    )
    return parser.parse_known_args(args)[0]


def start_ood_monitor(ood_cfg, static_dir):
    """Returns (OODMonitor, OODWebServer) or (None, None) when disabled/unbuilt.
    Imported lazily -- ood_monitor pulls in sklearn + umap, which the rest of this
    node doesn't need."""
    ood_cfg = ood_cfg or {}
    if not ood_cfg.get("enabled", False):
        # Say so explicitly: the config read here is the INSTALLED copy, so editing
        # src/.../topics.yaml without re-running colcon build looks like a dead page.
        print("[act_infer_mujoco] OOD monitor disabled (ood_monitor.enabled is false in the "
              "installed config -- edit config/topics.yaml, then re-run colcon build)")
        return None, None
    reference_dir = Path(ood_cfg.get("reference_dir", "")).expanduser()
    if not reference_dir.is_dir():
        print(f"[act_infer_mujoco] ood_monitor enabled but reference_dir not found: {reference_dir} -- skipping "
              f"(build it with `ros2 run policy_runner ood_build_reference`)")
        return None, None

    from policy_runner.ood_monitor import OODMonitor, OODWebServer

    monitor = OODMonitor(
        reference_dir, knn_k=int(ood_cfg.get("knn_k", 5)), umap_hz=float(ood_cfg.get("umap_hz", 10.0))
    )
    web_cfg = ood_cfg.get("web") or {}
    host = web_cfg.get("host", "0.0.0.0")
    port = int(web_cfg.get("port", 8081))
    web = OODWebServer(monitor, static_dir, host=host, port=port)
    web.start()
    print(f"[act_infer_mujoco] OOD monitor -> http://{host}:{port}/  (modalities: {', '.join(monitor.modalities)})")
    return monitor, web


def main(args=None):
    cli = parse_args(args)
    share = Path(get_package_share_directory("policy_runner"))
    config = load_config(cli.config or share / "config" / "topics.yaml")
    act_cfg = config.get("act_inference")
    if not act_cfg:
        raise ValueError("config must contain an `act_inference` section (see config/topics.yaml)")
    ckpt_path = cli.ckpt or act_cfg["ckpt_path"]
    control_hz = cli.hz or float(act_cfg.get("control_hz", 30.0))

    print(f"[act_infer_mujoco] loading policy from {ckpt_path} ...")
    policy = ACTChunkPolicy(
        ckpt_path,
        act_repo_root=act_cfg.get("act_repo_root", "~/act"),
        temporal_agg=bool(act_cfg.get("temporal_agg", False)),
        temporal_agg_k=float(act_cfg.get("temporal_agg_k", 0.01)),
        temporal_agg_newest=bool(act_cfg.get("temporal_agg_newest", False)),
    )
    print(
        f"[act_infer_mujoco] chunk_size={policy.chunk_size} camera_names={policy.camera_names} "
        f"device={policy.device} temporal_agg={policy.temporal_agg} "
        f"temporal_agg_newest={policy.temporal_agg_newest} action_repr={policy.action_repr}"
    )
    viz = MujocoQposViz(act_cfg["mjcf_path"], launch_viewer=not cli.no_viewer)

    if cli.rerun:
        init_rerun_joint_plots()

    ood, ood_web = start_ood_monitor(config.get("ood_monitor"), share / "static")

    udp_sender = None
    if not cli.no_udp:
        udp_cfg = act_cfg.get("udp_output", {}) or {}
        udp_host = cli.udp_host or udp_cfg.get("host", "")
        udp_port = cli.udp_port or int(udp_cfg.get("port", 0) or 0)
        if udp_host and udp_port:
            udp_sender = UdpJointSender(udp_host, udp_port, JOINT_ORDER)
            print(f"[act_infer_mujoco] UDP joint output -> {udp_host}:{udp_port}")
        else:
            print(
                "[act_infer_mujoco] UDP joint output enabled but no host/port configured -- skipping "
                "(set act_inference.udp_output in topics.yaml, or pass --udp-host/--udp-port)"
            )

    rclpy.init(args=args)
    node = InferInputNode(config, rerun_enabled=cli.rerun)
    executor = MultiThreadedExecutor(num_threads=max(4, len(node.camera_names) + 2))
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    period_s = 1.0 / control_hz
    action_idx = 0
    infer_idx = 0
    try:
        next_tick = time.monotonic()
        while rclpy.ok() and viz.is_running():
            obs = node.latest_observation()
            if obs is None:
                node.get_logger().warn("waiting for fresh camera + joint_states data...", throttle_duration_sec=2.0)
            else:
                qpos, images = obs
                action = policy.next_action(qpos, images)
                viz.set_qpos(action, shadow_qpos=qpos)  # shadow robot = real observed qpos
                if ood is not None:
                    # Cheap: just hands the sample to the monitor's worker thread. Image
                    # features only refresh on did_infer ticks; in between the monitor
                    # holds the last image-side score.
                    ood.update(qpos, policy.last_backbone_features if policy.did_infer else None)
                action_idx += 1  # chunk/output -> next action to send: every tick, refresh in place
                if udp_sender is not None:
                    # Clip to the MJCF joint limits before it leaves for the real robot -- viz above
                    # still shows the policy's raw output so an out-of-range action is visible.
                    clipped_action = np.clip(action, viz.joint_range[:, 0], viz.joint_range[:, 1])
                    udp_sender.send(clipped_action, sequence=action_idx)
                    if cli.rerun:
                        rr.set_time("wall_time", timestamp=time.time())
                        for name, position in zip(JOINT_ORDER, clipped_action):
                            rr.log(f"joints/{name}/target", rr.Scalars([float(position)]))
                if policy.did_infer:  # camera+qpos -> model query: rare, log as its own line
                    infer_idx += 1
                    print(f"\n[act_infer_mujoco] INFER #{infer_idx} (action #{action_idx})")
                print(f"\r[act_infer_mujoco] action #{action_idx:6d}", end="", flush=True)
            viz.sync()
            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()  # fell behind (e.g. first chunk query) -- don't try to catch up
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        viz.close()
        if ood_web is not None:
            ood_web.stop()
        if udp_sender is not None:
            udp_sender.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
