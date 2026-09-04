"""Offline: builds the OOD-monitor's training-set reference from the demonstration
hdf5 episodes -- run once per checkpoint, inside the container (needs the act repo +
the actual training data, neither of which live in this dev workspace).

For each of qpos and each camera's ACT backbone feature, saves the raw training
vectors (for runtime k-NN distance) and a fitted UMAP reducer + its 2D embedding of
the training set (for the live web page's background cloud + live-point projection).
Standard aloha/act hdf5 schema: /observations/qpos (T, state_dim), /observations/
images/<camera_name> (T, H, W, 3 uint8 RGB).

Usage (inside container):
  python3 -m policy_runner.ood_reference_builder \
      --data-dir ~/act/data/real_pick_yellow_bottle \
      --ckpt /path/to/policy_best.ckpt --out /path/to/ckpt_dir/ood_reference
"""

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import umap

from policy_runner.act_model import ACTChunkPolicy


def _load_episode(path, camera_names):
    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][()]
        images = {cam: f[f"observations/images/{cam}"][()] for cam in camera_names}
    return qpos, images


def build(data_dir, ckpt_path, out_dir, act_repo_root, stride, umap_neighbors, umap_min_dist):
    policy = ACTChunkPolicy(ckpt_path, act_repo_root=act_repo_root)
    episodes = sorted(Path(data_dir).expanduser().glob("*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"no .hdf5 episodes under {data_dir}")
    print(f"[ood_reference_builder] {len(episodes)} episodes, camera_names={policy.camera_names}, stride={stride}")

    qpos_rows = []
    feat_rows = {cam: [] for cam in policy.camera_names}
    for ep_path in episodes:
        qpos, images = _load_episode(ep_path, policy.camera_names)
        n_steps = len(qpos)
        for t in range(0, n_steps, stride):
            frame_images = {cam: images[cam][t] for cam in policy.camera_names}
            features = policy.query_backbone_features(qpos[t], frame_images)
            qpos_rows.append(qpos[t])
            for cam in policy.camera_names:
                feat_rows[cam].append(features[cam])
        print(f"  {ep_path.name}: {n_steps} steps -> {len(range(0, n_steps, stride))} sampled")

    qpos_arr = np.stack(qpos_rows).astype(np.float32)
    feat_arrs = {cam: np.stack(rows).astype(np.float32) for cam, rows in feat_rows.items()}
    print(f"[ood_reference_builder] reference set: {len(qpos_arr)} points, qpos_dim={qpos_arr.shape[1]}, "
          f"feat_dim={next(iter(feat_arrs.values())).shape[1]}")

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "data.npz", qpos=qpos_arr, **{f"feat_{cam}": arr for cam, arr in feat_arrs.items()})

    reducers = {}
    modalities = {"qpos": qpos_arr, **{cam: feat_arrs[cam] for cam in policy.camera_names}}
    for name, arr in modalities.items():
        print(f"[ood_reference_builder] fitting UMAP for '{name}' ({arr.shape}) ...")
        reducers[name] = umap.UMAP(
            n_components=2, n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=0
        ).fit(arr)
    with open(out_dir / "reducers.pkl", "wb") as f:
        pickle.dump(reducers, f)

    print(f"[ood_reference_builder] wrote {out_dir}/data.npz and {out_dir}/reducers.pkl")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="directory of training episode .hdf5 files")
    parser.add_argument("--ckpt", required=True, help="path to policy_best.ckpt")
    parser.add_argument("--out", required=True, help="output directory for data.npz + reducers.pkl")
    parser.add_argument("--act-repo-root", default="~/act")
    parser.add_argument("--stride", type=int, default=5, help="subsample every Nth frame per episode")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="umap.UMAP(n_neighbors=)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="umap.UMAP(min_dist=)")
    return parser.parse_args(args)


def main(args=None):
    cli = parse_args(args)
    build(cli.data_dir, cli.ckpt, cli.out, cli.act_repo_root, cli.stride, cli.umap_neighbors, cli.umap_min_dist)


if __name__ == "__main__":
    main()
