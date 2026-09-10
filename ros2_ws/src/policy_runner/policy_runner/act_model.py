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
        # dataset_stats.pkl loaded early: it's the authoritative source for joint_ids and
        # action_repr. config_hydra_resolved.yaml's own joint_ids/state_dim/action_dim are
        # null on every run that sets them via a plain-list CLI override (a Hydra resolution
        # quirk, not a "run didn't use them" signal) -- imitate_episodes.py itself only ever
        # derives state_dim/action_dim from joint_ids in-memory, never writes them back to the
        # resolved config, and its own eval/replay scripts (replay_eval.py, eval_common_horizon.py)
        # read joint_ids from dataset_stats.pkl for the same reason.
        with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "rb") as f:
            self.stats = pickle.load(f)
        # act/imitate_episodes.py trains some runs to predict action - qpos ("delta") instead
        # of the raw action ("absolute"); dataset_stats.pkl records which. Denormalizing a
        # delta checkpoint's output with action_mean/action_std (as if it were absolute)
        # silently produces a target near the training set's mean qpos, unrelated to the
        # robot's actual current position or the scene -- see act/eval_deploy_metrics.py's
        # denorm() / imitate_episodes.py's post_process for the reference implementation.
        self.action_repr = self.stats.get("action_repr", "absolute")

        task_config = TASK_CONFIGS[hydra_cfg["task_name"]]
        # camera_names is a per-run Hydra CLI override (config_hydra_resolved.yaml carries
        # the actual training cameras); TASK_CONFIGS only has the task's default, which
        # drifts across sweeps that override cameras under the same task_name.
        self.camera_names = hydra_cfg.get("camera_names") or task_config["camera_names"]
        # dataset_dir is a per-run Hydra CLI override too (same drift as camera_names above);
        # constants.py's DATA_DIR is relative to the act repo root, so make it absolute
        # so ood_reference_builder.py can default --data-dir to it from anywhere.
        dataset_dir = hydra_cfg.get("dataset_dir") or task_config["dataset_dir"]
        self.dataset_dir = os.path.join(os.path.expanduser(act_repo_root), dataset_dir)
        # (H, W) training resized every camera to before stacking (act/utils.py
        # load_cam_images) -- cameras differ in native resolution, so live inference
        # must match or np.stack fails. None means training didn't resize (cameras
        # already matched).
        self.image_size = hydra_cfg.get("image_size")
        # joint_ids: set when the checkpoint was trained on a subset of JOINT_NAMES (e.g. the
        # 8 arm+wrist joints, dropping the 16 finger joints -- see act/utils.py EpisodicDataset).
        joint_ids = self.stats.get("joint_ids")
        self.joint_ids = np.asarray(joint_ids, dtype=int) if joint_ids is not None else None
        # state_dim/action_dim: mirrors imitate_episodes.py's own derivation exactly. A
        # task_space + joint_ids run fuses the arm into the palm pose (POSE_DIM, always) and
        # keeps only the joint_ids-selected trailing wrist/finger dims on top (see
        # _task_space_keep_idx); joint_ids alone (no task_space) just narrows the raw joint
        # vector; neither is derivable from the (always-null, see above) hydra config.
        self.state_dim = hydra_cfg.get("state_dim") or task_config["state_dim"]
        self.action_dim = hydra_cfg.get("action_dim") or task_config.get("action_dim", self.state_dim)
        if self.action_repr == "task_space":
            self.state_dim = self.action_dim = (
                len(self._task_space_keep_idx(self.joint_ids)) if self.joint_ids is not None
                else forward_kinematics.TASKSPACE_DIM
            )
        elif self.joint_ids is not None:
            self.state_dim = self.action_dim = len(self.joint_ids)
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

        # task_space: next_action() returns pos(3)+quat_wxyz(4)+hand instead of joint targets
        # (see _post_action) -- these are the hand's joint names, in the order that trailing
        # block comes out in. joint_ids (arm-only) drops some of the 18 to just the ones kept
        # by _task_space_keep_idx; without joint_ids all 18 survive.
        if self.joint_ids is not None:
            self.hand_joint_names = [
                forward_kinematics.JOINT_NAMES[j] for j in self.joint_ids if j >= forward_kinematics.N_ARM
            ]
        else:
            self.hand_joint_names = list(forward_kinematics.JOINT_NAMES[forward_kinematics.N_ARM:])

        self.temporal_agg = temporal_agg
        self.temporal_agg_k = temporal_agg_k
        self.temporal_agg_newest = temporal_agg_newest

        self._chunk = None  # (chunk_size, action_dim) raw actions, buffered from the last query
        self._chunk_step = 0
        self._chunk_reference = None  # _reference_state(qpos) cached at the last query time
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

    def _task_space_keep_idx(self, joint_ids):
        """Mirrors act/utils.py's _task_space_keep_idx: the arm is always fused whole into the
        palm pose, so joint_ids only selects which of the trailing (POSE_DIM-onward) wrist/finger
        dims of the 27-dim FK'd state/action survive."""
        hand_keep = [j - self._fk.N_ARM for j in joint_ids if j >= self._fk.N_ARM]
        return np.array(list(range(self._fk.POSE_DIM)) + [self._fk.POSE_DIM + h for h in hand_keep], dtype=int)

    def _reference_state(self, qpos):
        """The value action/delta targets are expressed relative to: the FK'd task-space
        state (pos+rot6d+hand, 27-dim, trimmed by joint_ids if the checkpoint was trained on
        an arm-only subset) for task_space, else the plain robot-qpos slice (joint_ids subset,
        if any)."""
        qpos = np.asarray(qpos, dtype=np.float32)
        if self.action_repr == "task_space":
            state = self._fk.task_state(qpos)
            return state[self._task_space_keep_idx(self.joint_ids)] if self.joint_ids is not None else state
        if self.joint_ids is not None:
            return qpos[self.joint_ids]
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

    def _expand_joint_ids(self, action, qpos):
        """joint_ids checkpoints predict only those joints; every other joint (e.g. the 16
        finger joints on an arm+wrist-only checkpoint) holds at its current live qpos.
        Not applicable to task_space -- its action is already pos+quat+hand (see
        _post_action/hand_joint_names), not a vector indexed by joint_ids."""
        if self.joint_ids is None or self.action_repr == "task_space":
            return action
        full = np.asarray(qpos, dtype=np.float64).copy()
        full[self.joint_ids] = action
        return full

    def _query_chunk(self, qpos, images_by_camera):
        """Runs the model once, returns the raw (chunk_size, action_dim) predicted chunk."""
        qpos_raw = np.asarray(qpos, dtype=np.float32)
        if self.action_repr == "task_space":
            qpos_raw = self._fk.task_state(qpos_raw)  # 24-dim robot qpos -> 27-dim state
            if self.joint_ids is not None:
                qpos_raw = qpos_raw[self._task_space_keep_idx(self.joint_ids)]
        elif self.joint_ids is not None:
            qpos_raw = qpos_raw[self.joint_ids]
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
        """qpos: full (24,) array, JOINT_NAMES/JOINT_ORDER order -- always the whole robot,
        not just state_dim of it (task_space needs it for FK; joint_ids checkpoints need it
        to hold the untrained joints, e.g. fingers, at their current reading).
        images_by_camera: {camera_name: HxWx3 uint8 RGB ndarray}, one entry per
        self.camera_names. Returns raw (unnormalized) joint targets, (24,) -- joint_ids
        checkpoints have the predicted subset written into qpos's current reading (see
        _expand_joint_ids) -- except for task_space, which returns pos(3)+quat_wxyz(4)+hand
        (see hand_joint_names for the trailing block's joint order/length), no IK applied."""
        if self.temporal_agg:
            return self._next_action_temporal_agg(qpos, images_by_camera)
        return self._next_action_chunked(qpos, images_by_camera)

    def _next_action_chunked(self, qpos, images_by_camera):
        self.did_infer = self._chunk is None or self._chunk_step >= self.chunk_size
        if self.did_infer:
            self._chunk = self._query_chunk(qpos, images_by_camera)
            self._chunk_step = 0
            # Cache the reference at query time -- act/utils.py's delta target is
            # `action - qpos[start_ts]`, ONE reference broadcast over the whole chunk, not a
            # per-step one. Recomputing _reference_state(qpos) on every call (as this used to)
            # fed later chunk steps a reference that had drifted from the query-time qpos,
            # silently breaking the trained delta semantics after step 0 of every chunk.
            self._chunk_reference = self._reference_state(qpos)
        raw_action = self._chunk[self._chunk_step]
        self._chunk_step += 1
        action = self._post_action(raw_action, self._chunk_reference)
        return self._expand_joint_ids(action, qpos)

    def _next_action_temporal_agg(self, qpos, images_by_camera):
        """Query every step; average this step's predictions from every still-relevant
        buffered chunk, weighted by exp(-k * age). Upstream (`temporal_agg_newest=False`)
        counts age oldest-chunk-first, so the OLDEST chunk still covering this step gets the
        largest weight -- at 30Hz/chunk_size=50 that's a near-uniform average over ~1.7s of
        stale predictions, ~0.8s of lag. `temporal_agg_newest=True` reverses it (same fix as
        `act/imitate_episodes.py`'s `--temporal_agg_newest`): measured there at -47.5% mean
        command error with k=0.5, no retraining needed.
        Delta reconstruction (`_post_action`) happens once, after averaging, using qpos at
        the CURRENT step -- correct here because every step re-queries (query_frequency=1), so
        the current qpos IS that query's start reference; matches imitate_episodes.py's
        post_process(raw_action, chunk_reference_qpos), which is refreshed every step for the
        same reason. See `_next_action_chunked` for the non-temporal_agg case, where the
        reference must instead be cached and held fixed across a whole open-loop chunk."""
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
        action = self._post_action(raw_action, self._reference_state(qpos))
        return self._expand_joint_ids(action, qpos)
