"""Sends the policy's predicted joint targets to a LAN host over UDP as JSON,
for real-robot control. Fire-and-forget (no ack/retry) -- the receiver is
expected to just consume the latest packet each control tick.
"""

import json
import socket


class UdpJointSender:
    def __init__(self, host, port, joint_names):
        self.host = host
        self.port = port
        self.joint_names = list(joint_names)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, positions, sequence):
        payload = {
            "sequence": sequence,
            "positions": {name: float(p) for name, p in zip(self.joint_names, positions)},
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))

    def close(self):
        self.sock.close()


class UdpPoseHandSender:
    """Sends a task-space prediction (palm pos + quat_wxyz + hand joint positions) to a LAN
    host over UDP as JSON. No IK here -- the receiver is expected to solve for arm joint
    targets itself; this just reports the palm pose target and the hand joint targets
    directly. Fire-and-forget, same as UdpJointSender."""

    def __init__(self, host, port, hand_joint_names):
        self.host = host
        self.port = port
        self.hand_joint_names = list(hand_joint_names)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, t, sim_time, pos, quat_wxyz, hand_positions):
        payload = {
            "t": t,
            "sim_time": sim_time,
            "hand": "right",
            "wrist_pos": [float(v) for v in pos],
            "wrist_quat_wxyz": [float(v) for v in quat_wxyz],
            "joint_names": self.hand_joint_names,
            "joint_angles": [float(p) for p in hand_positions],
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))

    def close(self):
        self.sock.close()
