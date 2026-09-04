"""Runtime OOD indicator: per-tick k-NN distance (qpos + per-camera ACT backbone
feature) against the training reference set built offline by
ood_reference_builder.py, plus a UMAP-projected live point for the /ood/ web page.
Logging-only by design (see discussion) -- this computes and serves scores, it does
not threshold or gate anything.

Web server follows web_monitor.WebServer's plain ThreadingHTTPServer + polling
pattern (no websocket) for consistency with the rest of this package.
"""

import json
import pickle
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.neighbors import NearestNeighbors


class OODMonitor:
    def __init__(self, reference_dir, knn_k=5, umap_hz=3.0):
        reference_dir = Path(reference_dir)
        data = np.load(reference_dir / "data.npz")
        with open(reference_dir / "reducers.pkl", "rb") as f:
            self.reducers = pickle.load(f)  # {modality: fitted umap.UMAP}

        self.modalities = list(self.reducers.keys())  # e.g. ["qpos", "left", "top"]
        self._raw = {"qpos": data["qpos"]}
        self._raw.update({name: data[f"feat_{name}"] for name in self.modalities if name != "qpos"})
        self.knn = {
            name: NearestNeighbors(n_neighbors=min(knn_k, len(arr))).fit(arr) for name, arr in self._raw.items()
        }
        # Training-set cloud for the web page's background scatter; computed once here
        # rather than re-sent every poll.
        self.background = {name: self.reducers[name].embedding_.tolist() for name in self.modalities}

        self.umap_period_ns = int(1e9 / umap_hz)
        self._last_umap_ns = 0
        self.lock = threading.Lock()
        self.latest = {"t": None, "knn_dist": {}, "embed": {}}

    def update(self, qpos, backbone_features):
        """qpos: (state_dim,) raw robot-unit array, same units as the training hdf5
        qpos ood_reference_builder.py read -- not the policy's normalized qpos.
        backbone_features: {camera_name: (C,)} from policy.last_backbone_features, or
        None/{} on a tick where the policy didn't query (chunked mode, no
        temporal_agg) -- image-side scores just hold their last value that tick."""
        now_ns = time.time_ns()
        points = {"qpos": np.asarray(qpos, dtype=np.float32)}
        if backbone_features:
            points.update(backbone_features)

        knn_dist = dict(self.latest["knn_dist"])
        for name, vec in points.items():
            dist, _ = self.knn[name].kneighbors(vec[None, :])
            knn_dist[name] = float(dist.mean())

        embed = dict(self.latest["embed"])
        if now_ns - self._last_umap_ns >= self.umap_period_ns:  # UMAP transform is pricier -- rate-limit it
            self._last_umap_ns = now_ns
            for name, vec in points.items():
                xy = self.reducers[name].transform(vec[None, :])[0]
                embed[name] = [float(xy[0]), float(xy[1])]

        with self.lock:
            self.latest = {"t": now_ns, "knn_dist": knn_dist, "embed": embed}

    def status(self):
        with self.lock:
            return dict(self.latest)


class OODWebServer:
    def __init__(self, monitor, static_dir, host="0.0.0.0", port=8081):
        self.monitor = monitor
        static_dir = Path(static_dir)
        reference_payload = {"modalities": monitor.modalities, "background": monitor.background}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _safe_write(self, data):
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def _json(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self._safe_write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/ood/reference":
                    self._json(reference_payload)
                    return
                if path == "/ood/status":
                    self._json(outer.monitor.status())
                    return
                filename = "ood.html" if path in ("/", "/ood", "/ood/") else path.lstrip("/")
                if filename not in {"ood.html", "ood.js"}:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                target = static_dir / filename
                if not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = {".html": "text/html", ".js": "application/javascript"}[target.suffix]
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self._safe_write(data)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="ood-web", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
