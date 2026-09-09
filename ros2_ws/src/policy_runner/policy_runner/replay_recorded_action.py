"""Open-loop sanity check: replay a held-out episode's RECORDED action (ground truth from
the ACT hdf5 -- what the human/state-machine actually did -- NOT a model prediction)
straight to the real robot over UDP, at the dataset's native resample rate, with the
episode's own recorded camera video played back in lockstep (same frame index -- they were
already resampled onto one shared 30Hz grid by convert_teleop_dataset.py, so no separate
alignment step is needed here).

Purpose: isolate "is the deployment stack sane" (joint order, units, clipping, control
rate) from "is the checkpoint/training data any good". If the robot can't even reproduce a
recorded demonstration played back open-loop, the bug is in this stack (act_infer_mujoco.py
/ udp_joint_sender.py / the UDP receiver), not in act's training pipeline -- no amount of
sweeping hyperparameters there will fix it. See act/README.md and
act/results/sweep_20260908/REPORT.md for the training-side analysis this is meant to
rule in/out.

No camera subscription or live qpos is needed for the replay itself (the action and video
are already on disk) -- rclpy is only imported at all for the optional --shadow-live overlay.

    python3 replay_recorded_action.py --episode 3
    python3 replay_recorded_action.py --episode 3 --no-udp                # viewer-only dry run
    python3 replay_recorded_action.py --episode 3 --speed 0.5             # half speed (robot + video)
    python3 replay_recorded_action.py --episode 3 --cameras left top      # only these camera tiles
    python3 replay_recorded_action.py --episode 3 --shadow-live           # overlay real /joint_states
    python3 replay_recorded_action.py --episode 3 --dataset_dir /home/asu/act/data/real_pick_yellow_bottle/good_41_tw240
"""

import argparse
import threading
import time
from pathlib import Path

import h5py
import numpy as np
from ament_index_python.packages import get_package_share_directory

from policy_runner.mujoco_qpos_viz import JOINT_ORDER, MujocoQposViz
from policy_runner.udp_joint_sender import UdpJointSender
from policy_runner.web_monitor import load_config


def load_episode(dataset_dir, episode, cameras, start, end):
    """Recorded action[start:end] plus the same-range camera frames, from one open of
    episode_{episode}.hdf5 -- guarantees action and video are sliced identically (they're
    already frame-aligned by construction: convert_teleop_dataset.py resamples both onto the
    same 30Hz grid, so index i is the same instant in both). `cameras=None` loads every camera
    recorded in the episode; `cameras=False` skips video loading entirely."""
    path = Path(dataset_dir).expanduser() / f"episode_{episode}.hdf5"
    with h5py.File(path, "r") as root:
        action = root["/action"][()]
        if action.shape[1] != len(JOINT_ORDER):
            raise ValueError(
                f"{path}: action has {action.shape[1]} dims, expected {len(JOINT_ORDER)} (JOINT_ORDER) -- "
                f"point --dataset_dir at the full un-subsetted dataset (e.g. good_41 / good_41_tw240), "
                f"not a joint_ids-subset run's own dir"
            )
        end = end if end is not None else len(action)
        action = action[start:end]
        frames, cams = {}, []
        if cameras is not False:
            available = list(root["observations/images"].keys())
            cams = cameras or available
            missing = [c for c in cams if c not in available]
            if missing:
                raise ValueError(f"{path}: cameras {missing} not recorded here (has {available})")
            frames = {c: root[f"observations/images/{c}"][start:end] for c in cams}
    return action, frames, cams


def tile_frames(frame_by_cam, cams, tile_height=240):
    """Stack this tick's per-camera RGB frames into one BGR image for cv2.imshow, each
    resized to a common height so mismatched-resolution cameras (see README, 480x640 /
    480x848 / 400x400) still line up in one row."""
    import cv2

    tiles = []
    for c in cams:
        img = frame_by_cam[c]
        h, w = img.shape[:2]
        scale = tile_height / h
        tiles.append(cv2.resize(img, (round(w * scale), tile_height)))
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR)


def parse_args(args=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="", help="path to topics.yaml (default: installed config/topics.yaml)")
    p.add_argument("--dataset_dir", default="", help="ACT hdf5 dataset dir; default: "
                   "<act_inference.act_repo_root>/data/real_pick_yellow_bottle/good_41")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--start", type=int, default=0, help="first frame to replay")
    p.add_argument("--end", type=int, default=None, help="last frame (exclusive); default: whole episode")
    p.add_argument("--hz", type=float, default=0.0,
                   help="override act_inference.control_hz (default 30, the dataset's own resample rate -- "
                        "this is the recording rate, not a speed knob; use --speed for that)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier applied to BOTH the joint commands and the video "
                        "preview together (same frame stream, just paced faster/slower) -- 1.0 = "
                        "native rate; <1 slows both down (recommended for a first real-robot pass: "
                        "the position waypoints are unchanged so implied joint velocity scales down "
                        "too); >1 speeds both up (implied velocity scales UP correspondingly -- be "
                        "careful near robot velocity/torque limits)")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="camera names to preview (default: every camera recorded in the episode)")
    p.add_argument("--no-video", action="store_true", help="don't load/show the recorded camera video")
    p.add_argument("--udp-host", default="", help="override act_inference.udp_output.host")
    p.add_argument("--udp-port", type=int, default=0, help="override act_inference.udp_output.port")
    p.add_argument("--no-udp", action="store_true", help="viewer-only dry run, don't send to the robot")
    p.add_argument("--no-viewer", action="store_true", help="don't open the MuJoCo preview window")
    p.add_argument("--shadow-live", action="store_true",
                   help="subscribe /joint_states and show the real robot's actual pose as the "
                        "viewer's shadow overlay next to the recorded target (needs ROS + the robot up)")
    p.add_argument("--loop", action="store_true", help="repeat the episode until Ctrl-C")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt before sending to the robot")
    return p.parse_known_args(args)[0]


def main(args=None):
    cli = parse_args(args)
    share = Path(get_package_share_directory("policy_runner"))
    config = load_config(cli.config or share / "config" / "topics.yaml")
    act_cfg = config.get("act_inference")
    if not act_cfg:
        raise ValueError("config must contain an `act_inference` section (see config/topics.yaml)")
    control_hz = cli.hz or float(act_cfg.get("control_hz", 30.0))
    if cli.speed <= 0:
        raise ValueError(f"--speed must be > 0, got {cli.speed}")

    dataset_dir = cli.dataset_dir or (
        Path(act_cfg.get("act_repo_root", "~/act")).expanduser() / "data/real_pick_yellow_bottle/good_41"
    )
    show_video = not cli.no_video
    actions, frames, cams = load_episode(
        dataset_dir, cli.episode, False if not show_video else cli.cameras, cli.start, cli.end)
    print(f"[replay_recorded_action] episode {cli.episode}: {len(actions)} frames from {dataset_dir}"
          + (f", cameras {cams}" if show_video else ", video off"))

    # Always built (even with --no-viewer) for its joint_range -- the one safety net between
    # a recorded (or corrupted) value and the real actuators.
    viz = MujocoQposViz(act_cfg["mjcf_path_joint_space"], launch_viewer=not cli.no_viewer,
                        task_space=False, control_hz=control_hz)

    node = executor = None
    if cli.shadow_live:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        from policy_runner.act_infer_mujoco import InferInputNode
        rclpy.init(args=args)
        node = InferInputNode(config, camera_names=[])  # no cameras -- joint_states only
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()

    udp_sender = None
    if not cli.no_udp:
        udp_cfg = act_cfg.get("udp_output", {}) or {}
        udp_host = cli.udp_host or udp_cfg.get("host", "")
        udp_port = cli.udp_port or int(udp_cfg.get("port", 0) or 0)
        if udp_host and udp_port:
            udp_sender = UdpJointSender(udp_host, udp_port, JOINT_ORDER)
        else:
            print("[replay_recorded_action] no udp host/port configured -- viewer-only")

    if udp_sender is not None and not cli.yes:
        print(f"\n*** About to send {len(actions)} RECORDED frames straight to the real robot at "
              f"{udp_sender.host}:{udp_sender.port}, {control_hz * cli.speed:.1f} Hz "
              f"(native {control_hz:.0f} Hz x speed {cli.speed:g}). ***")
        if cli.speed > 1:
            print(f"    speed > 1: implied joint velocity is {cli.speed:g}x the recorded motion.")
        if input("Type 'yes' to proceed: ").strip().lower() != "yes":
            print("aborted")
            return

    period_s = 1.0 / (control_hz * cli.speed)
    try:
        seq = 0
        while True:
            next_tick = time.monotonic()
            for i, action in enumerate(actions):
                clipped = np.clip(action, viz.joint_range[:, 0], viz.joint_range[:, 1])
                shadow_qpos = None
                if node is not None:
                    obs = node.latest_observation()
                    shadow_qpos = obs[0] if obs is not None else None
                viz.set_qpos(clipped, shadow_qpos=shadow_qpos)
                viz.sync()
                if show_video:
                    import cv2
                    cv2.imshow("replay_recorded_action", tile_frames({c: frames[c][i] for c in cams}, cams))
                    cv2.waitKey(1)
                if udp_sender is not None:
                    udp_sender.send(clipped, sequence=seq)
                seq += 1
                print(f"\r[replay_recorded_action] frame {seq:6d}/{len(actions)}", end="", flush=True)
                if not viz.is_running():
                    return
                next_tick += period_s
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()  # fell behind -- don't try to catch up
            print()
            if not cli.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()
        if show_video:
            import cv2
            cv2.destroyAllWindows()
        if udp_sender is not None:
            udp_sender.close()
        if node is not None:
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
