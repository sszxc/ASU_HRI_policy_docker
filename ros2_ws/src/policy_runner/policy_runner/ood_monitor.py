"""Runtime OOD indicator: per-modality (qpos + one per camera backbone feature) k-NN
distance against the training reference set built by ood_reference_builder.py, plus a
UMAP projection of the live point for the browser page.

Logging/visualization only -- it computes and serves scores, it never thresholds,
alerts or gates the policy.

All the maths runs on a worker thread: update() only stashes the latest sample, so
the 30Hz control loop never pays for a k-NN query or a UMAP transform (the latter is
tens of ms and would otherwise drop control ticks).

The web server follows web_monitor.WebServer's plain ThreadingHTTPServer + polling
pattern (no websocket), for consistency with the rest of this package.
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
    def __init__(self, reference_dir, knn_k=5, umap_hz=10.0):
        reference_dir = Path(reference_dir)
        data = np.load(reference_dir / "data.npz")
        with open(reference_dir / "reducers.pkl", "rb") as f:
            bundle = pickle.load(f)  # unpickling the reducers needs umap importable

        self.modalities = list(bundle["modalities"])  # ["qpos", <camera names...>]
        self.reducers = bundle["reducers"]
        self.baseline = {name: {str(p): v for p, v in scores.items()} for name, scores in bundle["baseline"].items()}
        self.qpos_mean = data["qpos_mean"]
        self.qpos_std = data["qpos_std"]
        self.knn = {
            name: NearestNeighbors(n_neighbors=min(knn_k, len(data[f"ref_{name}"]))).fit(data[f"ref_{name}"])
            for name in self.modalities
        }
        # Training-set 2D cloud for the page's background scatter -- sent once, not per poll.
        self.background = {name: data[f"emb_{name}"].tolist() for name in self.modalities}

        self.umap_period_s = 1.0 / umap_hz if umap_hz > 0 else 0.0
        self._last_umap_s = 0.0
        self._lock = threading.Lock()
        self._pending = None  # newest unprocessed {modality: vector}; older samples are dropped
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.latest = {"t": None, "knn_dist": {}, "embed": {}}
        self.ready = False
        self._reference = {name: data[f"ref_{name}"][:1] for name in self.modalities}  # warm-up input only
        self._worker = threading.Thread(target=self._run, name="ood-worker", daemon=True)
        self._worker.start()

    def update(self, qpos, backbone_features=None):
        """Called from the control loop -- cheap, just hands off the newest sample.
        qpos: (state_dim,) raw robot-unit joint positions, same units as the training
        hdf5 (normalization happens here). backbone_features: {camera_name: (C,)} from
        policy.last_backbone_features, or None on a tick where the policy didn't query
        the model (chunked mode) -- the image-side scores then hold their last value."""
        point = {"qpos": (np.asarray(qpos, dtype=np.float32) - self.qpos_mean) / self.qpos_std}
        if backbone_features:
            point.update({k: np.asarray(v, dtype=np.float32) for k, v in backbone_features.items()})
        with self._lock:
            self._pending = point
        self._wake.set()

    def _run(self):
        # UMAP.transform's first call JITs (numba), several seconds for the whole set --
        # burn it here on the worker thread so the first live sample isn't swallowed.
        for name, vec in self._reference.items():
            self.reducers[name].transform(vec)
        self._reference = None
        self.ready = True
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                point, self._pending = self._pending, None
            if point is None:
                continue
            knn_dist = dict(self.latest["knn_dist"])
            for name, vec in point.items():
                dist, _ = self.knn[name].kneighbors(vec[None, :])
                knn_dist[name] = float(dist.mean())

            embed = dict(self.latest["embed"])
            now_s = time.monotonic()
            if now_s - self._last_umap_s >= self.umap_period_s:  # UMAP.transform is the expensive part
                self._last_umap_s = now_s
                for name, vec in point.items():
                    xy = self.reducers[name].transform(vec[None, :])[0]
                    embed[name] = [float(xy[0]), float(xy[1])]

            with self._lock:
                self.latest = {"t": time.time(), "knn_dist": knn_dist, "embed": embed}

    def status(self):
        with self._lock:
            return dict(self.latest)

    def stop(self):
        self._stop.set()
        self._wake.set()


class OODWebServer:
    def __init__(self, monitor, static_dir, host="0.0.0.0", port=8081):
        self.monitor = monitor
        static_dir = Path(static_dir)
        reference_payload = {
            "modalities": monitor.modalities,
            "background": monitor.background,
            "baseline": monitor.baseline,
        }
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
                if filename not in {"ood.html", "ood.js", "styles.css"}:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                target = static_dir / filename
                if not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}[target.suffix]
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
        self.monitor.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
