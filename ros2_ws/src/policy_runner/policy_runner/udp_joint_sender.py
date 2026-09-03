"""Sends the policy's predicted joint targets to a LAN host over UDP as JSON,
for real-robot control. Fire-and-forget (no ack/retry) -- the receiver is
expected to just consume the latest packet each control tick.
"""

import json
import socket
import time


class UdpJointSender:
    def __init__(self, host, port, joint_names):
        self.host = host
        self.port = port
        self.joint_names = list(joint_names)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, positions):
        payload = {
            "timestamp": time.time(),
            "joint_names": self.joint_names,
            "positions": [float(p) for p in positions],
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))

    def close(self):
        self.sock.close()
