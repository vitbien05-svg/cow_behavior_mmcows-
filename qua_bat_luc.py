"""
merge_dataset.py
================
Merge sensor (.npy) và image features (.pt) theo (cow_id, timestamp),
sau đó gom thành sequences non-overlapping với độ dài T configurable.

Sensor format : (N, 16, 258)
    - dim [:256] = features
    - dim [256]  = cow_id
    - dim [257]  = timestamp

Image format  : dict .pt với keys:
    - 'features'  : Tensor (256, 8, 8) mỗi sample
    - 'cow_id'    : int/Tensor
    - 'timestamp'  : int/Tensor
    - 'behavior'  : int/Tensor  (0..7, bỏ label 0, remap 1..7 -> 0..6)

Output:
    sensor  : (B, T, 256)       float32
    image   : (B, T, 256, 8, 8) float32
    labels  : (B,)              int64   — majority vote trong chunk
    cow_ids : (B,)              int64   — để object-wise split
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _to_int(val) -> int:
    """Convert tensor / numpy scalar / python int sang int thuần."""
    if isinstance(val, torch.Tensor):
        return int(val.item())
    if isinstance(val, np.ndarray):
        return int(val.flat[0])
    return int(val)


def _majority_vote(labels: list[int]) -> int:
    """Trả về nhãn xuất hiện nhiều nhất trong chunk."""
    return Counter(labels).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SENSOR
# ─────────────────────────────────────────────────────────────────────────────


def load_sensor(sensor_path: str | Path) -> dict[tuple[int, int], np.ndarray]:
    """
    Load sensor .npy và tách ra từng frame riêng lẻ.

    Returns
    -------
    sensor_map : dict[(cow_id, timestamp)] -> feature array (256,)
    """
    arr = np.load(sensor_path, allow_pickle=False)  # (N, 16, 258)

    if arr.ndim == 3:
        N, T_orig, D = arr.shape
        assert D == 258, f"Expected dim 258, got {D}"
        arr = arr.reshape(N * T_orig, D)  # (N*16, 258)
    elif arr.ndim == 2:
        # Đã được flatten sẵn từ trước
        assert arr.shape[1] == 258, f"Expected dim 258, got {arr.shape[1]}"
    else:
        raise ValueError(f"Unexpected sensor ndim={arr.ndim}")

    features = arr[:, :256]  # (M, 256)
    cow_ids = arr[:, 256].astype(int)
    timestamps = arr[:, 257].astype(int)

    sensor_map: dict[tuple[int, int], np.ndarray] = {}
    for feat, cid, ts in zip(features, cow_ids, timestamps):
        key = (cid, ts)
        # Nếu trùng key: giữ lần đầu (hoặc có thể overwrite tuỳ yêu cầu)
        if key not in sensor_map:
            sensor_map[key] = feat.astype(np.float32)

    print(f"[Sensor] Loaded {len(sensor_map):,} unique (cow_id, timestamp) frames")
    return sensor_map


# ─────────────────────────────────────────────────────────────────────────────
# LOAD IMAGE
# ─────────────────────────────────────────────────────────────────────────────


def load_image(image_path: str | Path) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
    """
    Load image dict .pt.

    Returns
    -------
    image_map : dict[(cow_id, timestamp)] -> (feature (256,8,8), label_remapped)
        - label 0 bị bỏ
        - label 1..7 remap -> 0..6
    """
    data = torch.load(image_path, map_location="cpu", weights_only=False)

    # Hỗ trợ cả list-of-dict lẫn dict-of-lists
    if isinstance(data, (list, tuple)):
        records = data
    elif isinstance(data, dict):
        # dict-of-lists: {'cow_id': [...], 'timestep': [...], ...}
        keys = list(data.keys())
        n = len(data[keys[0]])
        records = [{k: data[k][i] for k in keys} for i in range(n)]
    else:
        raise ValueError(f"Unsupported image data type: {type(data)}")

    image_map: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    skipped_label0 = 0

    for rec in records:
        behavior = _to_int(rec["behavior"])

        # Bỏ label 0
        if behavior == 0:
            skipped_label0 += 1
            continue

        # Remap 1..7 -> 0..6
        label = behavior - 1

        cid = _to_int(rec["cow_id"])
        ts = _to_int(rec["timestamp"])

        feat = rec["features"]
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        feat = feat.astype(np.float32)  # (256, 8, 8)
        assert feat.shape == (256, 8, 8), f"Unexpected feature shape: {feat.shape}"

        key = (cid, ts)
        if key not in image_map:
            image_map[key] = (feat, label)

    print(
        f"[Image]  Loaded {len(image_map):,} unique (cow_id, timestamp) frames "
        f"(skipped {skipped_label0} label-0 samples)"
    )
    return image_map


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────


def merge(
    sensor_map: dict[tuple[int, int], np.ndarray],
    image_map: dict[tuple[int, int], tuple[np.ndarray, int]],
) -> dict[int, list[tuple[int, np.ndarray, np.ndarray, int]]]:
    """
    Inner join theo (cow_id, timestamp).

    Returns
    -------
    per_cow : dict[cow_id] -> list of (timestamp, sensor_feat, image_feat, label)
              Đã sort theo timestamp tăng dần.
    """
    common_keys = set(sensor_map.keys()) & set(image_map.keys())
    print(f"[Merge]  Common keys: {len(common_keys):,} frames")

    per_cow: dict[int, list] = {}
    for cid, ts in common_keys:
        s_feat = sensor_map[(cid, ts)]  # (256,)
        i_feat, label = image_map[(cid, ts)]  # (256,8,8), int

        if cid not in per_cow:
            per_cow[cid] = []
        per_cow[cid].append((ts, s_feat, i_feat, label))

    # Sort theo timestamp tăng dần cho từng con bò
    for cid in per_cow:
        per_cow[cid].sort(key=lambda x: x[0])

    cow_list = sorted(per_cow.keys())
    total_frames = sum(len(v) for v in per_cow.values())
    print(f"[Merge]  {len(cow_list)} cows, {total_frames:,} total frames")
    return per_cow


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK INTO SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────


def build_sequences(
    per_cow: dict[int, list],
    # T: int = 16,
    T: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gom từng con bò thành các chunk non-overlapping, pad chunk cuối nếu thiếu.

    Parameters
    ----------
    per_cow : output của merge()
    T       : sequence length (configurable: 8, 12, 16, ...)

    Returns
    -------
    sensor  : (B, T, 256)       float32
    image   : (B, T, 256, 8, 8) float32
    labels  : (B,)              int64
    cow_ids : (B,)              int64
    """
    all_sensor = []
    all_image = []
    all_labels = []
    all_cow_ids = []

    # Zero-pad tensors
    PAD_SENSOR = np.zeros((256,), dtype=np.float32)
    PAD_IMAGE = np.zeros((256, 8, 8), dtype=np.float32)

    for cid, frames in sorted(per_cow.items()):
        # frames: list of (ts, s_feat, i_feat, label)
        n_frames = len(frames)

        # Chia thành các chunk kích thước T
        for start in range(0, n_frames, T):
            chunk = frames[start : start + T]
            actual_len = len(chunk)

            s_chunk = [f[1] for f in chunk]  # list of (256,)
            i_chunk = [f[2] for f in chunk]  # list of (256,8,8)
            l_chunk = [f[3] for f in chunk]  # list of int labels

            # Pad nếu chunk cuối thiếu frame
            if actual_len < T:
                pad_n = T - actual_len
                s_chunk += [PAD_SENSOR] * pad_n
                i_chunk += [PAD_IMAGE] * pad_n
                # Label pad không tham gia majority vote

            # Majority vote chỉ trên actual frames (không tính pad)
            label = _majority_vote(l_chunk)

            all_sensor.append(np.stack(s_chunk, axis=0))  # (T, 256)
            all_image.append(np.stack(i_chunk, axis=0))  # (T, 256, 8, 8)
            all_labels.append(label)
            all_cow_ids.append(cid)

    # Stack tất cả thành batch
    sensor_tensor = torch.from_numpy(np.stack(all_sensor, axis=0))  # (B, T, 256)
    image_tensor = torch.from_numpy(np.stack(all_image, axis=0))  # (B, T, 256, 8, 8)
    labels_tensor = torch.tensor(all_labels, dtype=torch.int64)  # (B,)
    cow_ids_tensor = torch.tensor(all_cow_ids, dtype=torch.int64)  # (B,)

    B = sensor_tensor.shape[0]
    print(f"\n[Build]  T={T} | Sequences built: {B}")
    print(f"         sensor  : {tuple(sensor_tensor.shape)}")
    print(f"         image   : {tuple(image_tensor.shape)}")
    print(
        f"         labels  : {tuple(labels_tensor.shape)}  unique={labels_tensor.unique().tolist()}"
    )
    print(
        f"         cow_ids : {tuple(cow_ids_tensor.shape)} unique cows={cow_ids_tensor.unique().numel()}"
    )

    return sensor_tensor, image_tensor, labels_tensor, cow_ids_tensor


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(
    sensor_path: str | Path,
    image_path: str | Path,
    T: int = 16,
    save_path: str | Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full pipeline: load -> merge -> chunk -> (optionally) save.

    Parameters
    ----------
    sensor_path : path to sensor .npy file
    image_path  : path to image dict .pt file
    T           : sequence length
    save_path   : nếu không None, lưu output ra file .pt

    Returns
    -------
    sensor, image, labels, cow_ids
    """
    print("=" * 60)
    print(f"Building dataset  T={T}")
    print("=" * 60)

    sensor_map = load_sensor(sensor_path)
    image_map = load_image(image_path)
    per_cow = merge(sensor_map, image_map)
    sensor, image, labels, cow_ids = build_sequences(per_cow, T=T)

    if save_path is not None:
        save_path = Path(save_path)
        torch.save(
            {
                "sensor": sensor,
                "image": image,
                "labels": labels,
                "cow_ids": cow_ids,
                "T": T,
            },
            save_path,
        )
        print(f"\n[Save]   Dataset saved -> {save_path}")

    print("=" * 60)
    return sensor, image, labels, cow_ids


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK (chạy với data giả)
# ─────────────────────────────────────────────────────────────────────────────


def _run_sanity_check():
    """Tạo fake data để verify pipeline end-to-end."""
    import tempfile

    print("\n[Sanity Check] Running with fake data...")

    RNG = np.random.default_rng(42)
    N_COW = 5
    T_MAX = 30  # mỗi con bò có 30 frame (để test cả pad case)
    T_SEQ = 12

    # ── Fake sensor ──────────────────────────────────────────────────────────
    records = []
    for cid in range(1, N_COW + 1):
        for ts in range(T_MAX):
            feat = RNG.random(256, dtype=np.float32)
            row = np.concatenate([feat, [float(cid), float(ts)]])  # (258,)
            records.append(row)

    # Bọc thành (N, 16, 258) như BiLSTM output thật
    records_arr = np.array(records)  # (N_COW*T_MAX, 258)
    # Pad lên bội số của 16 để reshape được
    total_rows = records_arr.shape[0]
    pad_rows = (16 - total_rows % 16) % 16
    if pad_rows > 0:
        records_arr = np.vstack(
            [records_arr, np.zeros((pad_rows, 258), dtype=np.float32)]
        )
    sensor_arr = records_arr.reshape(-1, 16, 258)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, sensor_arr)
        sensor_tmp = f.name

    # ── Fake image ───────────────────────────────────────────────────────────
    img_records = []
    for cid in range(1, N_COW + 1):
        for ts in range(T_MAX):
            behavior = RNG.integers(1, 8)  # label 1..7 (bỏ 0)
            feat = RNG.random((256, 8, 8), dtype=np.float32)
            img_records.append(
                {
                    "cow_id": cid,
                    "timestamp": ts,
                    "behavior": int(behavior),
                    "features": torch.from_numpy(feat),
                }
            )
    # Thêm vài record label=0 để test bỏ
    for cid in range(1, 3):
        for ts in range(T_MAX, T_MAX + 5):
            img_records.append(
                {
                    "cow_id": cid,
                    "timestamp": ts,
                    "behavior": 0,
                    "features": torch.zeros(256, 8, 8),
                }
            )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(img_records, f.name)
        image_tmp = f.name

    # ── Run pipeline ─────────────────────────────────────────────────────────
    sensor, image, labels, cow_ids = build_dataset(
        sensor_path=sensor_tmp,
        image_path=image_tmp,
        T=T_SEQ,
    )

    # Assertions
    B, T, D = sensor.shape
    Bi, Ti, C, H, W = image.shape

    assert T == T_SEQ, f"T mismatch: {T} vs {T_SEQ}"
    assert D == 256, f"Sensor dim mismatch: {D}"
    assert C == 256 and H == 8 and W == 8, f"Image shape mismatch"
    assert B == Bi == len(labels) == len(cow_ids)
    assert (
        labels.min() >= 0 and labels.max() <= 6
    ), f"Label out of range: {labels.min()}..{labels.max()}"
    assert cow_ids.unique().numel() == N_COW

    print(f"\n✓ All assertions passed.")
    print(f"  sensor  : {tuple(sensor.shape)}")
    print(f"  image   : {tuple(image.shape)}")
    print(f"  labels  : {tuple(labels.shape)}  range=[{labels.min()},{labels.max()}]")
    print(f"  cow_ids : {tuple(cow_ids.shape)}")

    # Cleanup
    os.unlink(sensor_tmp)
    os.unlink(image_tmp)


if __name__ == "__main__":
    sensor_path = (
        r"D:\nhung kien thuc dai hoc\Semester 05\DPL\mmcow_git\sensor_combined_3d.npy"
    )
    image_path = r"D:\nhung kien thuc dai hoc\Semester 05\DPL\mmcow_git\image_features_minimal_unix.pt"
    save_path = (
        r"D:\nhung kien thuc dai hoc\Semester 05\DPL\mmcow_git\fusion_dataset_T16.pt"
    )

    build_dataset(
        sensor_path=sensor_path,
        image_path=image_path,
        T=16,
        save_path=save_path,
    )
