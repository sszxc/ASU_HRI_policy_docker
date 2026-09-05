"""Offline: build the OOD monitor's training-set reference from the demonstration
hdf5 episodes. Run once per checkpoint, inside the container (needs the act repo,
the checkpoint and the training data).

For qpos and for each camera's ACT backbone feature it saves:
  - the raw training vectors, for the runtime k-NN distance;
  - a fitted UMAP reducer + the training set's own 2D embedding, for the live page's
    background cloud and the live point's projection;
  - the training set's own leave-one-out k-NN distance percentiles, so the live
    number can be read as a ratio against in-distribution data instead of a bare
    float (see ood_monitor.py / static/ood.js).

Usage (inside the container):
  ros2 run policy_runner ood_build_reference --ckpt <.../policy_best.ckpt>
Defaults: --data-dir from the checkpoint's own task config, --out <ckpt_dir>/ood_reference.
"""

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors

from policy_runner.act_model import ACTChunkPolicy


def _episode_frames(path, camera_names, stride):
    """Reads only the strided frames -- episodes here run ~1600 steps x 480x640x3 per
    camera, so slurping a whole episode's image dataset is several GB."""
    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][()]
        idx = list(range(0, len(qpos), stride))
        images = {cam: f[f"observations/images/{cam}"][idx] for cam in camera_names}
    return qpos[idx], images


def _loo_knn_percentiles(arr, knn_k):
    """Leave-one-out mean k-NN distance within the training set itself -> percentiles.
    Query with k+1 neighbours and drop the first (the point itself, distance 0)."""
    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, len(arr))).fit(arr)
    dist, _ = nn.kneighbors(arr)
    loo = dist[:, 1:].mean(axis=1)
    return {p: float(np.percentile(loo, p)) for p in (50, 95, 99)}


def build(data_dir, ckpt_path, out_dir, act_repo_root, stride, knn_k, umap_neighbors, umap_min_dist):
    import umap  # heavy (numba); imported here so --help stays instant

    policy = ACTChunkPolicy(ckpt_path, act_repo_root=act_repo_root)
    data_dir = Path(data_dir or policy.dataset_dir).expanduser()
    episodes = sorted(data_dir.glob("*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"no .hdf5 episodes under {data_dir}")
    print(f"[ood_reference] {len(episodes)} episodes under {data_dir}, "
          f"camera_names={policy.camera_names}, stride={stride}")

    qpos_rows = []
    feat_rows = {cam: [] for cam in policy.camera_names}
    for ep_i, ep_path in enumerate(episodes):
        qpos, images = _episode_frames(ep_path, policy.camera_names, stride)
        for t in range(len(qpos)):
            features = policy.query_backbone_features(qpos[t], {c: images[c][t] for c in policy.camera_names})
            qpos_rows.append(qpos[t])
            for cam in policy.camera_names:
                feat_rows[cam].append(features[cam])
        print(f"  [{ep_i + 1}/{len(episodes)}] {ep_path.name}: +{len(qpos)} samples "
              f"(total {len(qpos_rows)})", flush=True)

    qpos_arr = np.stack(qpos_rows).astype(np.float32)
    feat_arrs = {cam: np.stack(rows).astype(np.float32) for cam, rows in feat_rows.items()}
    # qpos joints have wildly different ranges; standardize with the checkpoint's own
    # dataset stats so the k-NN distance isn't dominated by the widest-swinging joint.
    # ood_monitor.py applies the same transform to the live qpos.
    qpos_mean = np.asarray(policy.stats["qpos_mean"], dtype=np.float32).reshape(-1)[: qpos_arr.shape[1]]
    qpos_std = np.asarray(policy.stats["qpos_std"], dtype=np.float32).reshape(-1)[: qpos_arr.shape[1]]
    modalities = {"qpos": (qpos_arr - qpos_mean) / qpos_std}
    modalities.update({cam: feat_arrs[cam] for cam in policy.camera_names})
    print(f"[ood_reference] {len(qpos_arr)} samples; "
          + ", ".join(f"{n}={a.shape[1]}d" for n, a in modalities.items()))

    reducers, embeddings, baseline = {}, {}, {}
    for name, arr in modalities.items():
        baseline[name] = _loo_knn_percentiles(arr, knn_k)
        print(f"[ood_reference] fitting UMAP for '{name}' {arr.shape} "
              f"(in-distribution k-NN p50/p95={baseline[name][50]:.3f}/{baseline[name][95]:.3f}) ...", flush=True)
        reducer = umap.UMAP(
            n_components=2, n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=0
        ).fit(arr)
        reducers[name] = reducer
        embeddings[name] = reducer.embedding_.astype(np.float32)

    out_dir = Path(out_dir or Path(ckpt_path).parent / "ood_reference").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "data.npz",
        qpos_mean=qpos_mean,
        qpos_std=qpos_std,
        knn_k=np.int32(knn_k),
        **{f"ref_{name}": arr for name, arr in modalities.items()},
        **{f"emb_{name}": emb for name, emb in embeddings.items()},
    )
    with open(out_dir / "reducers.pkl", "wb") as f:
        pickle.dump({"modalities": list(modalities), "reducers": reducers, "baseline": baseline}, f)
    print(f"[ood_reference] wrote {out_dir}/data.npz + reducers.pkl -- "
          f"point ood_monitor.reference_dir at {out_dir}")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Build the OOD monitor's training-set reference.")
    parser.add_argument("--ckpt", required=True, help="path to policy_best.ckpt")
    parser.add_argument("--data-dir", default="", help="training episode .hdf5 dir (default: the ckpt's task config)")
    parser.add_argument("--out", default="", help="output dir (default: <ckpt_dir>/ood_reference)")
    parser.add_argument("--act-repo-root", default="~/act")
    parser.add_argument("--stride", type=int, default=10, help="keep every Nth frame per episode")
    parser.add_argument("--knn-k", type=int, default=5, help="k for the distance baseline; match ood_monitor.knn_k")
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser.parse_known_args(args)[0]


def main(args=None):
    cli = parse_args(args)
    build(cli.data_dir, cli.ckpt, cli.out, cli.act_repo_root, cli.stride, cli.knn_k,
          cli.umap_neighbors, cli.umap_min_dist)


if __name__ == "__main__":
    main()
