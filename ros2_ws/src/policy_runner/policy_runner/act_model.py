"""Load an ACT checkpoint and run chunked inference for a live control loop.

Mirrors `act/imitate_episodes.py`'s `eval_bc()` loading path exactly (task config from
`aloha_scripts/constants.py` via the checkpoint's own `config_hydra_resolved.yaml`,
weights from the `.ckpt`, normalization from `dataset_stats.pkl`) and both of its
action-chunking modes, selected by `temporal_agg`:
- off (default): query the model once every `chunk_size` calls, serve the buffered
  chunk open-loop in between.
- on (`--temporal_agg` in the original repo): query every step; blend the current
  step's overlapping predictions from recent chunks with an exponentially decaying
  weight (older chunks weighted more) -- smoother/more consistent, at chunk_size-x
  inference cost.
"""

import os
import pickle
import sys
from collections import deque

import cv2
import numpy as np
import torch


def _add_act_repo_to_syspath(act_repo_root):
    act_repo_root = os.path.expanduser(act_repo_root)
    if act_repo_root not in sys.path:
        sys.path.insert(0, act_repo_root)
    detr_path = os.path.join(act_repo_root, "detr")
    if os.path.isdir(detr_path) and detr_path not in sys.path:
        sys.path.insert(0, detr_path)


class ACTChunkPolicy:
    def __init__(self, ckpt_path, act_repo_root="~/act", device=None, temporal_agg=False,
                 temporal_agg_k=0.01, temporal_agg_newest=False):
        _add_act_repo_to_syspath(act_repo_root)
        import yaml
        from aloha_scripts.constants import TASK_CONFIGS
        from policy import ACTPolicy
        import forward_kinematics
        self._fk = forward_kinematics

        ckpt_dir = os.path.dirname(ckpt_path)
        with open(os.path.join(ckpt_dir, "config_hydra_resolved.yaml")) as f:
            hydra_cfg = yaml.safe_load(f)

        task_config = TASK_CONFIGS[hydra_cfg["task_name"]]
        # camera_names is a per-run Hydra CLI override (config_hydra_resolved.yaml carries
        # the actual training cameras); TASK_CONFIGS only has the task's default, which
        # drifts across sweeps that override cameras under the same task_name.
        self.camera_names = hydra_cfg.get("camera_names") or task_config["camera_names"]
        # constants.py's DATA_DIR is relative to the act repo root; make it absolute
        # so ood_reference_builder.py can default --data-dir to it from anywhere.
        self.dataset_dir = os.path.join(os.path.expanduser(act_repo_root), task_config["dataset_dir"])
        # (H, W) training resized every camera to before stacking (act/utils.py
        # load_cam_images) -- cameras differ in native resolution, so live inference
        # must match or np.stack fails. None means training didn't resize (cameras
        # already matched).
        self.image_size = hydra_cfg.get("image_size")
        # state_dim/action_dim: prefer the run's own resolved config over TASK_CONFIGS -- a
        # task_space run overrides these via CLI (state_dim=27 action_dim=27, see act/README),
        # which TASK_CONFIGS (fixed per task_name) has no way to reflect.
        self.state_dim = hydra_cfg.get("state_dim") or task_config["state_dim"]
        self.action_dim = hydra_cfg.get("action_dim") or task_config.get("action_dim", self.state_dim)
        self.chunk_size = hydra_cfg["chunk_size"]

        policy_config = {
            "lr": hydra_cfg["lr"],
            "num_queries": self.chunk_size,
            "kl_weight": hydra_cfg["kl_weight"],
            "hidden_dim": hydra_cfg["hidden_dim"],
            "dim_feedforward": hydra_cfg["dim_feedforward"],
            "latent_z_dim": hydra_cfg.get("latent_z_dim", 32),
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": self.camera_names,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ACTPolicy(policy_config)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        status = self.policy.load_state_dict(state_dict, strict=False)
        if status.missing_keys or status.unexpected_keys:
            print(
                f"[act_model] load_state_dict: missing={status.missing_keys} "
                f"unexpected={status.unexpected_keys}"
            )
        self.policy.to(self.device)
        self.policy.eval()

        with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "rb") as f:
            self.stats = pickle.load(f)
        # act/imitate_episodes.py trains some runs to predict action - qpos ("delta") instead
        # of the raw action ("absolute"); dataset_stats.pkl records which. Denormalizing a
        # delta checkpoint's output with action_mean/action_std (as if it were absolute)
        # silently produces a target near the training set's mean qpos, unrelated to the
        # robot's actual current position or the scene -- see act/eval_deploy_metrics.py's
        # denorm() / imitate_episodes.py's post_process for the reference implementation.
        self.action_repr = self.stats.get("action_repr", "absolute")
        # task_space: next_action() returns pos(3)+quat_wxyz(4)+hand(18) instead of joint
        # targets (see _post_action) -- these are the hand's 18 joint names, in the order
        # that trailing block comes out in.
        self.hand_joint_names = list(forward_kinematics.JOINT_NAMES[forward_kinematics.N_ARM:])

        self.temporal_agg = temporal_agg
        self.temporal_agg_k = temporal_agg_k
        self.temporal_agg_newest = temporal_agg_newest

        self._chunk = None  # (chunk_size, action_dim) raw actions, buffered from the last query
        self._chunk_step = 0
        self._chunk_buffer = deque(maxlen=self.chunk_size)  # temporal_agg only: [(start_step, chunk), ...]
        self._step = 0  # temporal_agg only: running query counter

        self.did_infer = False  # set by next_action(): True if this call queried the model

        # OOD indicator: one forward hook on the image backbone appends each camera's
        # globally-pooled feature during every self.policy(...) forward; _query_chunk
        # zips them back onto camera_names. This ACT fork shares a SINGLE backbone
        # (detr_vae.py: `self.backbones[0](image[:, cam_id])  # HARDCODED`) called once
        # per camera in camera_names order, so call order -- not module index -- is what
        # identifies the camera.
        self.last_backbone_features = {}  # {camera_name: (C,) np.ndarray}, refreshed per forward
        self._feature_buffer = []
        self.policy.model.backbones[0].register_forward_hook(self._backbone_hook)

    def _backbone_hook(self, _module, _inputs, output):
        features, _pos = output  # Joiner.forward -> ([last-layer (B,C,H,W)], [pos])
        pooled = features[0].mean(dim=(-2, -1))  # (B, C, H, W) -> (B, C)
        self._feature_buffer.append(pooled[0].detach().float().cpu().numpy())

    def _pre_qpos(self, qpos):
        return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]

    def _reference_state(self, qpos):
        """The value action/delta targets are expressed relative to: the FK'd task-space
        state (pos+rot6d+hand, 27-dim) for task_space, else the plain robot-qpos slice."""
        qpos = np.asarray(qpos, dtype=np.float32)
        if self.action_repr == "task_space":
            return self._fk.task_state(qpos)
        return qpos[: self.action_dim]

    def _post_action(self, action, qpos_raw):
        if self.action_repr == "task_space":
            # qpos_raw is _reference_state's FK'd state, not raw robot qpos. pos/hand deltas
            # add directly; the rotation delta is a 6D encoding of a *relative* rotation, so
            # it composes onto the reference by matrix multiply, not addition.
            delta = action * self.stats["task_space_std"] + self.stats["task_space_mean"]
            R0 = self._fk.sixd_to_rotmat(qpos_raw[3:9])
            R1 = R0 @ self._fk.sixd_to_rotmat(delta[3:9])
            pos = qpos_raw[:3] + delta[:3]
            hand = qpos_raw[9:] + delta[9:]
            return np.concatenate([pos, self._fk.rotmat_to_quat_wxyz(R1), hand])
        if self.action_repr == "delta":
            return action * self.stats["delta_std"] + self.stats["delta_mean"] + qpos_raw
        return action * self.stats["action_std"] + self.stats["action_mean"]

    def _query_chunk(self, qpos, images_by_camera):
        """Runs the model once, returns the raw (chunk_size, action_dim) predicted chunk."""
        qpos_raw = np.asarray(qpos, dtype=np.float32)
        if self.action_repr == "task_space":
            qpos_raw = self._fk.task_state(qpos_raw)  # 24-dim robot qpos -> 27-dim state
        else:
            qpos_raw = qpos_raw[: self.state_dim]
        qpos_n = self._pre_qpos(qpos_raw)
        qpos_t = torch.from_numpy(qpos_n.astype(np.float32)).float().to(self.device).unsqueeze(0)
        images = [images_by_camera[cam] for cam in self.camera_names]
        if self.image_size is not None:
            h, w = int(self.image_size[0]), int(self.image_size[1])
            images = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in images]
        image_np = np.stack([np.transpose(im, (2, 0, 1)) for im in images], axis=0)
        image_t = torch.from_numpy(image_np / 255.0).float().to(self.device).unsqueeze(0)
        self._feature_buffer.clear()
        all_actions = self.policy(qpos_t, image_t)
        self.last_backbone_features = dict(zip(self.camera_names, self._feature_buffer))
        return all_actions.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def query_backbone_features(self, qpos, images_by_camera):
        """Runs one forward pass purely for last_backbone_features, without touching
        next_action()'s chunk-buffer state. Used by ood_reference_builder.py; the live
        loop reads policy.last_backbone_features, which next_action() already refreshes."""
        self._query_chunk(qpos, images_by_camera)
        return self.last_backbone_features

    @torch.inference_mode()
    def next_action(self, qpos, images_by_camera):
        """qpos: (state_dim,) array, robot joint order (JOINT_NAMES/JOINT_ORDER for
        task_space -- FK needs the actual joint layout, not just state_dim of it).
        images_by_camera: {camera_name: HxWx3 uint8 RGB ndarray}, one entry per
        self.camera_names. Returns raw (unnormalized) joint targets, (action_dim,) --
        except for task_space, which returns pos(3)+quat_wxyz(4)+hand(18) (see
        hand_joint_names for the trailing block's joint order), no IK applied."""
        if self.temporal_agg:
            return self._next_action_temporal_agg(qpos, images_by_camera)
        return self._next_action_chunked(qpos, images_by_camera)

    def _next_action_chunked(self, qpos, images_by_camera):
        self.did_infer = self._chunk is None or self._chunk_step >= self.chunk_size
        if self.did_infer:
            self._chunk = self._query_chunk(qpos, images_by_camera)
            self._chunk_step = 0
        raw_action = self._chunk[self._chunk_step]
        self._chunk_step += 1
        return self._post_action(raw_action, self._reference_state(qpos))

    def _next_action_temporal_agg(self, qpos, images_by_camera):
        """Query every step; average this step's predictions from every still-relevant
        buffered chunk, weighted by exp(-k * age). Upstream (`temporal_agg_newest=False`)
        counts age oldest-chunk-first, so the OLDEST chunk still covering this step gets the
        largest weight -- at 30Hz/chunk_size=50 that's a near-uniform average over ~1.7s of
        stale predictions, ~0.8s of lag. `temporal_agg_newest=True` reverses it (same fix as
        `act/imitate_episodes.py`'s `--temporal_agg_newest`): measured there at -47.5% mean
        command error with k=0.5, no retraining needed.
        Delta reconstruction (`_post_action`) happens once, after averaging, using qpos at
        the CURRENT step -- matches imitate_episodes.py's post_process(raw_action, qpos_numpy)."""
        self.did_infer = True  # always queries -- kept for a uniform did_infer check regardless of mode
        chunk = self._query_chunk(qpos, images_by_camera)
        step = self._step
        self._chunk_buffer.append((step, chunk))
        self._step += 1

        preds = [c[step - start] for start, c in self._chunk_buffer if start <= step < start + self.chunk_size]
        order = np.arange(len(preds))[::-1] if self.temporal_agg_newest else np.arange(len(preds))
        weights = np.exp(-self.temporal_agg_k * order)
        weights /= weights.sum()
        raw_action = np.sum(np.stack(preds) * weights[:, None], axis=0)
        return self._post_action(raw_action, self._reference_state(qpos))
