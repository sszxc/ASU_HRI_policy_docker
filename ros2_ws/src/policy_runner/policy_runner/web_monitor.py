"""Subscribe camera images + joint_states over ROS2 and serve a browser
page showing which streams are live -- online/offline, last-seen age, and
message rate per topic. No recording/saving; this is purely a data-
availability check before running training/inference against a rig.

Page (static/index.html + app.js + styles.css) and the HTTP server below
are adapted from `trajectory_recorder/{node.py,static/}`
(`/home/asu/Downloads/trajectory_recorder_new/`), with the episode
recording (Start/Finish/label, HDF5/MP4, SessionWriter) stripped out and
camera grouping/joint streams driven by `config/topics.yaml` instead of
hardcoded names, since this rig's topics don't match that package's.
"""

import argparse
import json
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, JointState

_RAW_COLOR_ENCODINGS = {
    # encoding -> (dtype, channels, cv2 conversion to BGR, or None if already BGR)
    "bgr8": (np.uint8, 3, None),
    "rgb8": (np.uint8, 3, cv2.COLOR_RGB2BGR),
    "bgra8": (np.uint8, 4, cv2.COLOR_BGRA2BGR),
    "rgba8": (np.uint8, 4, cv2.COLOR_RGBA2BGR),
    "mono8": (np.uint8, 1, cv2.COLOR_GRAY2BGR),
    "8uc1": (np.uint8, 1, cv2.COLOR_GRAY2BGR),
}


def decode_raw_color(msg):
    """sensor_msgs/Image -> HxWx3 uint8 BGR ndarray (BGR since the only
    consumer here is cv2.imencode(".jpg", ...) for the browser preview)."""
    encoding = msg.encoding.lower()
    if encoding not in _RAW_COLOR_ENCODINGS:
        raise ValueError(f"unsupported raw color encoding: {msg.encoding}")
    dtype, channels, conversion = _RAW_COLOR_ENCODINGS[encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_values = msg.step // dtype.itemsize
    image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_values)
    image = image[:, : msg.width * channels]
    image = (
        image.reshape(msg.height, msg.width, channels)
        if channels > 1
        else image.reshape(msg.height, msg.width)
    )
    return cv2.cvtColor(image, conversion) if conversion is not None else image.copy()


def decode_compressed_color(data):
    image = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode compressed color image")
    return image


def jpeg(image, quality=80):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config.get("cameras"), dict) or not config["cameras"]:
        raise ValueError("config must contain at least one entry under `cameras`")
    if not isinstance(config.get("joint_states"), dict) or not config["joint_states"]:
        raise ValueError("config must contain at least one entry under `joint_states`")
    config.setdefault("image_transport", "raw")
    config.setdefault("preview_fps", 10.0)
    config.setdefault("online_timeout_seconds", 3.0)
    web = config.setdefault("web", {})
    web.setdefault("host", "0.0.0.0")
    web.setdefault("port", 8080)
    for spec in config["joint_states"].values():
        qos = spec.setdefault("qos", {})
        qos.setdefault("reliability", "reliable")
        qos.setdefault("depth", 10)
    return config


_RATE_WINDOW = 30  # samples kept per stream to estimate message rate


class _StreamStatus:
    """Online/last-seen/rate bookkeeping shared by camera and joint streams."""

    def __init__(self, topic):
        self.topic = topic
        self.last_seen_ns = None
        self.error = None
        self.recent_ns = deque(maxlen=_RATE_WINDOW)
        self.extra = {}

    def mark(self, now_ns):
        self.last_seen_ns = now_ns
        self.error = None
        self.recent_ns.append(now_ns)

    def snapshot(self, now_ns, timeout_ns):
        online = bool(self.last_seen_ns and now_ns - self.last_seen_ns < timeout_ns)
        hz = None
        if len(self.recent_ns) >= 2:
            span_s = (self.recent_ns[-1] - self.recent_ns[0]) / 1e9
            if span_s > 0:
                hz = (len(self.recent_ns) - 1) / span_s
        return {
            "topic": self.topic,
            "online": online,
            "last_seen_ns": self.last_seen_ns,
            "age_ms": (now_ns - self.last_seen_ns) / 1e6 if self.last_seen_ns else None,
            "hz": round(hz, 1) if hz is not None else None,
            "error": self.error,
            **self.extra,
        }


class MonitorNode(Node):
    def __init__(self, config):
        super().__init__("policy_runner_web_monitor")
        self.config = config
        self.image_transport = config["image_transport"]
        self.preview_interval_ns = int(1_000_000_000 / float(config["preview_fps"]))
        self.online_timeout_ns = int(float(config["online_timeout_seconds"]) * 1e9)
        self.lock = threading.Lock()
        self.latest_jpeg = {}
        self.last_preview_ns = {}
        self.streams = {}
        self._callback_groups = []

        for key, spec in config["cameras"].items():
            topic = spec["topic"]
            if self.image_transport == "compressed":
                topic = topic if topic.endswith("/compressed") else topic.rstrip("/") + "/compressed"
                message_type = CompressedImage
            else:
                message_type = Image
            self.streams[key] = _StreamStatus(topic)
            callback = lambda msg, key=key: self._on_image(key, msg)
            # Own MutuallyExclusiveCallbackGroup per subscription -- with the
            # default (shared) group, MultiThreadedExecutor still serializes
            # every camera behind whichever one is decoding, so the others'
            # best-effort frames get dropped by DDS while it waits.
            group = MutuallyExclusiveCallbackGroup()
            self._callback_groups.append(group)
            self.create_subscription(message_type, topic, callback, qos_profile_sensor_data, callback_group=group)
            self.get_logger().info(f"subscribed camera '{key}' ({spec['group']}) -> {topic}")

        for key, spec in config["joint_states"].items():
            topic = spec["topic"]
            self.streams[key] = _StreamStatus(topic)
            reliability = {
                "reliable": ReliabilityPolicy.RELIABLE,
                "best_effort": ReliabilityPolicy.BEST_EFFORT,
            }.get(str(spec["qos"]["reliability"]).lower())
            if reliability is None:
                raise ValueError(f"joint_states.{key}.qos.reliability must be reliable or best_effort")
            qos = QoSProfile(
                reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=int(spec["qos"]["depth"])
            )
            callback = lambda msg, key=key: self._on_joint_state(key, msg)
            group = MutuallyExclusiveCallbackGroup()
            self._callback_groups.append(group)
            self.create_subscription(JointState, topic, callback, qos, callback_group=group)
            self.get_logger().info(f"subscribed joint stream '{key}' -> {topic}")

    def _on_image(self, key, msg):
        now_ns = time.time_ns()
        with self.lock:
            self.streams[key].mark(now_ns)
            should_preview = now_ns - self.last_preview_ns.get(key, 0) >= self.preview_interval_ns
            if should_preview:
                self.last_preview_ns[key] = now_ns
        if not should_preview:
            return
        try:
            image = decode_raw_color(msg) if self.image_transport == "raw" else decode_compressed_color(msg.data)
            frame = jpeg(image)
            error = None
        except Exception as exc:
            frame = None
            error = str(exc)
        with self.lock:
            self.streams[key].error = error
            if frame is not None:
                self.latest_jpeg[key] = frame

    def _on_joint_state(self, key, msg):
        now_ns = time.time_ns()
        with self.lock:
            stream = self.streams[key]
            stream.mark(now_ns)
            stream.extra["joint_count"] = len(msg.name)

    def status(self):
        now_ns = time.time_ns()
        with self.lock:
            return {
                "server_time_ns": now_ns,
                "streams": {key: s.snapshot(now_ns, self.online_timeout_ns) for key, s in self.streams.items()},
            }


class WebServer:
    def __init__(self, node, config, static_dir):
        self.node = node
        self.static_dir = Path(static_dir)
        outer = self
        page_config = {
            "cameras": config["cameras"],
            "joint_states": {key: {"topic": spec["topic"]} for key, spec in config["joint_states"].items()},
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                outer.node.get_logger().debug(fmt % args)

            def _safe_write(self, data):
                try:
                    self.wfile.write(data)
                    return True
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return False

            def _json(self, payload, status=HTTPStatus.OK):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self._safe_write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/api/config":
                    self._json(page_config)
                    return
                if path == "/api/status":
                    self._json(outer.node.status())
                    return
                if path.startswith("/snapshot/") and path.endswith(".jpg"):
                    key = Path(path).stem
                    with outer.node.lock:
                        frame = outer.node.latest_jpeg.get(key)
                    if frame is None:
                        self.send_response(HTTPStatus.NO_CONTENT)
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self._safe_write(frame)
                    return
                filename = "index.html" if path == "/" else path.lstrip("/")
                if filename not in {"index.html", "styles.css", "app.js"}:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                target = outer.static_dir / filename
                if not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = {".html": "text/html", ".css": "text/css", ".js": "application/javascript"}.get(
                    target.suffix, "application/octet-stream"
                )
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self._safe_write(data)

        self.httpd = ThreadingHTTPServer((config["web"]["host"], int(config["web"]["port"])), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="monitor-web", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="path to topics.yaml (default: installed config/topics.yaml)")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--raw", dest="image_transport", action="store_const", const="raw")
    transport.add_argument("--compressed", dest="image_transport", action="store_const", const="compressed")
    parser.set_defaults(image_transport=None)
    return parser.parse_known_args(args)[0]


def main(args=None):
    cli = parse_args(args)
    share = Path(get_package_share_directory("policy_runner"))
    config = load_config(cli.config or share / "config" / "topics.yaml")
    if cli.image_transport:
        config["image_transport"] = cli.image_transport
    if cli.host:
        config["web"]["host"] = cli.host
    if cli.port:
        config["web"]["port"] = cli.port

    rclpy.init(args=args)
    node = MonitorNode(config)
    web = WebServer(node, config, share / "static")
    web.start()
    node.get_logger().info(f"web interface: http://{config['web']['host']}:{config['web']['port']}")
    executor = MultiThreadedExecutor(num_threads=min(16, max(4, len(node.streams))))
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        web.stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
