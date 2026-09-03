"""ACT inference node: subscribes the `left`/`top` camera + joint_states topics the
`real_pick_yellow_bottle` checkpoint was trained on, runs ACT at a fixed control_hz,
and writes the policy's raw joint-target output straight into a MuJoCo model's qpos
for visualization, alongside a translucent "shadow" duplicate of the robot driven by
the real observed qpos (the policy's own input) for a real-vs-target overlay -- no
UDP output, no actuators/mj_step, no real-robot control.

This is the inference half of README "Next phase" with the real-robot-output stage
swapped for a MuJoCo viewer (a visualize-before-you-command dev step). Camera/
joint_states subscription follows `web_monitor.py`'s pattern (per-subscription
MutuallyExclusiveCallbackGroup on a MultiThreadedExecutor, so one topic decoding
never blocks another). The ROS executor spins in a background thread; the main
thread owns the fixed-rate inference/viewer loop, since MuJoCo's passive viewer
wants one consistent thread calling `sync()`.
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState

from policy_runner.act_model import ACTChunkPolicy
from policy_runner.image_codec import decode_compressed_color
from policy_runner.mujoco_qpos_viz import MujocoQposViz
from policy_runner.web_monitor import load_config


class InferInputNode(Node):
    """Subscribes just the two training cameras + joint_states and caches the latest
    of each; the timed loop in main() reads it via latest_observation()."""

    def __init__(self, config):
        super().__init__("act_infer_mujoco")
        act_cfg = config["act_inference"]
        self.camera_names = act_cfg["camera_names"]
        self.online_timeout_ns = int(float(config["online_timeout_seconds"]) * 1e9)

        self.lock = threading.Lock()
        self.latest_image = {}
        self.latest_image_ns = {}
        self.latest_qpos = None
        self.latest_qpos_ns = None
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
        with self.lock:
            self.latest_qpos = np.array(msg.position, dtype=np.float32)
            self.latest_qpos_ns = time.time_ns()

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
    return parser.parse_known_args(args)[0]


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
    )
    print(
        f"[act_infer_mujoco] chunk_size={policy.chunk_size} camera_names={policy.camera_names} "
        f"device={policy.device} temporal_agg={policy.temporal_agg}"
    )
    viz = MujocoQposViz(act_cfg["mjcf_path"], launch_viewer=not cli.no_viewer)

    rclpy.init(args=args)
    node = InferInputNode(config)
    executor = MultiThreadedExecutor(num_threads=max(4, len(node.camera_names) + 2))
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    period_s = 1.0 / control_hz
    step = 0
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
                status = "INFER" if policy.did_infer else "cache"
                print(f"\r[act_infer_mujoco] step={step:6d} {status}", end="", flush=True)
                step += 1
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
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
