"""One-off diagnostic: does live inference's qpos INPUT source (`/joint_states`, subscribed by
act_infer_mujoco's InferInputNode) actually agree with the raw topics training labels were built
from (`trajectories/ur_arm` + `trajectories/robot_hand` in the recorder's trajectory.h5, whose
`topic` attrs give the exact source topic strings below)?

Why this and not something else: dataset replay (recorded ACTION -> UDP -> robot) has already
been confirmed to move the robot correctly, and act/results/.../deploy_metrics.json shows strong
offline track_corr (~0.85) on the exact "deployed path" (z=0 prior, images+qpos, no ground-truth
action) using the RECORDED qpos as input. Both of those validate the action/label side of the
pipeline. Neither touches this one link: the LIVE qpos fed to the policy at inference time comes
from a different topic (`/joint_states`, published by some other node) than the raw topics the
recorder itself combines into training qpos/action. If `/joint_states` disagrees with those raw
topics for any joint (offset, sign, scale, or a differently-ordered `name` list this script's
own by-name reindex doesn't already protect against), the model sees a "current state" it was
never trained on -- easily enough to make a policy with a good offline track_corr move the wrong
way live, without any of the above checks catching it.

Usage (inside the docker container, ROS + robot up):
    python3 diag_qpos_source.py
Prints a per-joint diff table once per second: /joint_states value vs. the matching raw-topic
value, for every joint both sides name. A joint with a consistently large or sign-flipped diff is
the bug; small (~sensor noise) diffs on every joint mean this link is fine and the problem is
elsewhere (camera framing/domain gap is the next thing to check).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState

# From trajectory.h5's trajectories/{ur_arm,robot_hand} `topic` attrs (authoritative -- these are
# the exact topics the recorder read to build training qpos/action, not a guess).
UR_ARM_TOPIC = "/ur/ur_joint_state_broadcaster/joint_states"
ROBOT_HAND_TOPIC = "/right_hand_finger_controller/hmf_actual/joint_states"
JOINT_STATES_TOPIC = "/joint_states"  # what act_infer_mujoco's InferInputNode actually subscribes to


class DiagNode(Node):
    def __init__(self):
        super().__init__("diag_qpos_source")
        self.latest = {}  # topic -> {name: position}
        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        for topic, qos in (
            (UR_ARM_TOPIC, qos_profile_sensor_data),
            (ROBOT_HAND_TOPIC, qos_profile_sensor_data),
            (JOINT_STATES_TOPIC, qos_reliable),
        ):
            self.create_subscription(JointState, topic, lambda msg, t=topic: self._on_msg(t, msg), qos)
        self.create_timer(1.0, self._print_diff)

    def _on_msg(self, topic, msg):
        self.latest[topic] = dict(zip(msg.name, msg.position))

    def _print_diff(self):
        js = self.latest.get(JOINT_STATES_TOPIC)
        if js is None:
            print("[diag] no /joint_states received yet")
            return
        print(f"\n{'joint':32s} {'source':10s} {'/joint_states':>14s} {'raw_topic':>14s} {'diff':>10s}")
        for src_topic, label in ((UR_ARM_TOPIC, "ur_arm"), (ROBOT_HAND_TOPIC, "robot_hand")):
            raw = self.latest.get(src_topic)
            if raw is None:
                print(f"  (no data yet on {src_topic})")
                continue
            for name, raw_val in raw.items():
                js_val = js.get(name)
                if js_val is None:
                    print(f"{name:32s} {label:10s} {'MISSING':>14s} {raw_val:14.4f} {'--':>10s}")
                    continue
                diff = js_val - raw_val
                flag = "  <-- CHECK" if abs(diff) > 0.02 else ""
                print(f"{name:32s} {label:10s} {js_val:14.4f} {raw_val:14.4f} {diff:10.4f}{flag}")


def main():
    rclpy.init()
    node = DiagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
