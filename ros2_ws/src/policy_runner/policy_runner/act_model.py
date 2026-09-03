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
    def __init__(self, ckpt_path, act_repo_root="~/act", device=None, temporal_agg=False, temporal_agg_k=0.01):
        _add_act_repo_to_syspath(act_repo_root)
        import yaml
        from aloha_scripts.constants import TASK_CONFIGS
        from policy import ACTPolicy

        ckpt_dir = os.path.dirname(ckpt_path)
        with open(os.path.join(ckpt_dir, "config_hydra_resolved.yaml")) as f:
            hydra_cfg = yaml.safe_load(f)

        task_config = TASK_CONFIGS[hydra_cfg["task_name"]]
        self.camera_names = task_config["camera_names"]
        self.state_dim = task_config["state_dim"]
        self.action_dim = task_config.get("action_dim", self.state_dim)
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

        self.temporal_agg = temporal_agg
        self.temporal_agg_k = temporal_agg_k

        self._chunk = None  # (chunk_size, action_dim) raw actions, buffered from the last query
        self._chunk_step = 0
        self._chunk_buffer = deque(maxlen=self.chunk_size)  # temporal_agg only: [(start_step, chunk), ...]
        self._step = 0  # temporal_agg only: running query counter

        self.did_infer = False  # set by next_action(): True if this call queried the model

    def _pre_qpos(self, qpos):
        return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]

    def _post_action(self, action):
        return action * self.stats["action_std"] + self.stats["action_mean"]

    def _query_chunk(self, qpos, images_by_camera):
        """Runs the model once, returns the raw (chunk_size, action_dim) predicted chunk."""
        qpos_n = self._pre_qpos(np.asarray(qpos, dtype=np.float32)[: self.state_dim])
        qpos_t = torch.from_numpy(qpos_n).float().to(self.device).unsqueeze(0)
        images = [images_by_camera[cam] for cam in self.camera_names]
        image_np = np.stack([np.transpose(im, (2, 0, 1)) for im in images], axis=0)
        image_t = torch.from_numpy(image_np / 255.0).float().to(self.device).unsqueeze(0)
        all_actions = self.policy(qpos_t, image_t)
        return all_actions.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def next_action(self, qpos, images_by_camera):
        """qpos: (state_dim,) array, robot joint order. images_by_camera: {camera_name:
        HxWx3 uint8 RGB ndarray}, one entry per self.camera_names. Returns (action_dim,)
        raw (unnormalized) joint targets."""
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
        return self._post_action(raw_action)

    def _next_action_temporal_agg(self, qpos, images_by_camera):
        """Query every step; average this step's predictions from every still-relevant
        buffered chunk, weighted by exp(-k * age) with age counted oldest-chunk-first
        (older chunks get the larger weight) -- same scheme as `act/imitate_episodes.py`
        eval_bc()'s `--temporal_agg` path."""
        self.did_infer = True  # always queries -- kept for a uniform did_infer check regardless of mode
        chunk = self._query_chunk(qpos, images_by_camera)
        step = self._step
        self._chunk_buffer.append((step, chunk))
        self._step += 1

        preds = [c[step - start] for start, c in self._chunk_buffer if start <= step < start + self.chunk_size]
        weights = np.exp(-self.temporal_agg_k * np.arange(len(preds)))
        weights /= weights.sum()
        raw_action = np.sum(np.stack(preds) * weights[:, None], axis=0)
        return self._post_action(raw_action)
