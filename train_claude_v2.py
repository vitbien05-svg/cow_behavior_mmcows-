"""
train.py
========
Training pipeline cho CowBehaviorModel — theo Object-wise Split (OS)
trong paper MMCOWS (NeurIPS 2024).

Flow:
    1. Load dataset từ file .pt
    2. Vẽ phân phối nhãn từng con bò (EDA)
    3. 5-fold cross-validation theo object-wise split:
         - 10 bò ghép 5 cặp cố định
         - Mỗi fold: 1 cặp = test, 1 cặp = val, 3 cặp còn lại = train
         - Xoay vòng qua 5 fold
    4. Gom prediction từ 5 fold → confusion matrix merged
    5. Vẽ đầy đủ: per-fold CM, merged CM, per-class F1, loss curves,
       phân phối nhãn, split distribution

Metric chính: F1 macro (không bị bias bởi class majority như lying ~53%)

Usage:
    python train.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, Subset

import sys

sys.path.append(str(Path(__file__).parent))
from model_layers import CowBehaviorModel

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_NAMES = [
    "Walking",
    "Standing",
    "Feed up",
    "Feed down",
    "Licking",
    "Drinking",
    "Lying",
]
# Màu cho từng behavior — dùng nhất quán trong tất cả plots
BEHAVIOR_COLORS = [
    "#378ADD",  # walking   — blue
    "#1D9E75",  # standing  — teal
    "#EF9F27",  # feed up   — amber
    "#D85A30",  # feed down — coral
    "#7F77DD",  # licking   — purple
    "#D4537E",  # drinking  — pink
    "#444441",  # lying     — gray
]
# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────


class CowDataset(Dataset):
    def __init__(
        self,
        sensor: torch.Tensor,  # (B, T, 256)
        image: torch.Tensor,  # (B, T, 256, 8, 8)
        labels: torch.Tensor,  # (B,)
        cow_ids: torch.Tensor,  # (B,)
        T: int = 16,
    ):
        self.sensor = sensor
        self.image = image
        self.labels = labels
        self.cow_ids = cow_ids
        self.T = T

        zero_frames = sensor.abs().sum(dim=-1) == 0  # (B, T)
        self.pad_mask = zero_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sensor[idx],  # (T, 256)
            self.image[idx],  # (T, 256, 8, 8)
            self.labels[idx],  # scalar
            self.pad_mask[idx],  # (T,) bool
        )


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT-WISE SPLIT — 5-FOLD theo paper MMCOWS
# ─────────────────────────────────────────────────────────────────────────────


def build_folds(
    cow_ids: torch.Tensor,
    n_folds: int = 5,
) -> list[dict]:
    """
    Ghép 10 bò thành 5 cặp cố định, xoay vòng qua 5 fold.

    Mỗi fold trả về dict:
        {
          'fold':      int,
          'train_idx': list[int],
          'val_idx':   list[int],
          'test_idx':  list[int],
          'train_cows': list,
          'val_cows':   list,
          'test_cows':  list,
        }

    Cấu trúc mỗi fold (paper MMCOWS):
        - test : 1 cặp  (2 bò chưa từng thấy trong fold này)
        - val  : 1 cặp
        - train: 3 cặp còn lại

    Fold i:
        test  = pair[i]
        val   = pair[(i+1) % n_folds]
        train = 3 cặp còn lại
    """
    unique_cows = sorted(cow_ids.unique().tolist())
    n_cows = len(unique_cows)

    if n_cows != 10:
        raise ValueError(
            f"Object-wise split theo paper yêu cầu đúng 10 bò, "
            f"nhưng dataset có {n_cows} bò: {unique_cows}. "
            f"Kiểm tra lại cow_ids trong file .pt."
        )

    # 5 cặp cố định: (bò 0,1), (bò 2,3), ..., (bò 8,9)
    pairs = [(unique_cows[2 * i], unique_cows[2 * i + 1]) for i in range(n_folds)]

    cow_ids_list = cow_ids.tolist()

    def ids_for_cows(cows):
        cow_set = set(cows)
        return [i for i, c in enumerate(cow_ids_list) if c in cow_set]

    folds = []
    for fold_idx in range(n_folds):
        test_pair = pairs[fold_idx]
        val_pair = pairs[(fold_idx + 1) % n_folds]
        train_pairs = [
            pairs[j]
            for j in range(n_folds)
            if j != fold_idx and j != (fold_idx + 1) % n_folds
        ]
        train_cows = [c for pair in train_pairs for c in pair]

        fold = {
            "fold": fold_idx + 1,
            "train_cows": sorted(train_cows),
            "val_cows": sorted(val_pair),
            "test_cows": sorted(test_pair),
            "train_idx": ids_for_cows(train_cows),
            "val_idx": ids_for_cows(val_pair),
            "test_idx": ids_for_cows(test_pair),
        }
        folds.append(fold)

    # In tóm tắt
    print("\n[Folds]  Object-wise 5-fold split (theo MMCOWS paper)")
    print(f"         Cặp cố định: {pairs}")
    print(f"  {'Fold':>4} | {'Train cows':>30} | {'Val':>8} | {'Test':>8}")
    print(f"  {'-'*60}")
    for f in folds:
        print(
            f"  {f['fold']:>4} | "
            f"{str(f['train_cows']):>30} | "
            f"{str(f['val_cows']):>8} | "
            f"{str(f['test_cows']):>8}"
        )

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = 7,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1)
    weights = labels.numel() / (num_classes * counts)
    return weights.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for sensor, image, labels, pad_mask in loader:
        sensor = sensor.to(device)
        image = image.to(device)
        labels = labels.to(device)
        pad_mask = pad_mask.to(device)

        optimizer.zero_grad()
        mask = pad_mask if pad_mask.any() else None
        logits = model(sensor, image, src_key_padding_mask=mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (avg_loss, accuracy, macro_f1, all_preds, all_labels)"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for sensor, image, labels, pad_mask in loader:
        sensor = sensor.to(device)
        image = image.to(device)
        labels = labels.to(device)
        pad_mask = pad_mask.to(device)

        mask = pad_mask if pad_mask.any() else None
        logits = model(sensor, image, src_key_padding_mask=mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    # ← F1 MACRO: mỗi class đóng góp ngang nhau, không bị bias bởi lying ~53%
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Áp dụng style tối giản nhất quán cho tất cả subplot."""
    ax.set_facecolor("#FAFAF8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCBC4")
    ax.tick_params(colors="#555554", labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", color="#2C2C2A", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color="#555554")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color="#555554")
    ax.grid(axis="y", color="#E0DFD8", linewidth=0.5, zorder=0)


# ── 1. Phân phối nhãn từng con bò ────────────────────────────────────────────


def plot_label_distribution_per_cow(
    cow_ids: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    save_dir: Path,
):
    """
    1 figure lớn, 10 subplot (2 hàng × 5 cột).
    Mỗi subplot = stacked bar chart phân phối hành vi của 1 con bò.
    """
    unique_cows = sorted(cow_ids.unique().tolist())
    n_cows = len(unique_cows)
    ncols = 5
    nrows = (n_cows + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.2, nrows * 3.0),
        facecolor="white",
    )
    fig.suptitle(
        "Phân phối hành vi theo từng con bò",
        fontsize=13,
        fontweight="bold",
        color="#2C2C2A",
        y=1.01,
    )
    axes_flat = axes.flatten() if n_cows > 1 else [axes]

    for ax_idx, cow_id in enumerate(unique_cows):
        ax = axes_flat[ax_idx]
        mask = cow_ids == cow_id
        cow_labels = labels[mask].tolist()
        total = len(cow_labels)

        counts = [cow_labels.count(c) for c in range(num_classes)]
        pcts = [c / total * 100 if total > 0 else 0 for c in counts]

        # Stacked horizontal bar
        left = 0.0
        for cls_idx, pct in enumerate(pcts):
            if pct == 0:
                continue
            ax.barh(
                0,
                pct,
                left=left,
                height=0.5,
                color=BEHAVIOR_COLORS[cls_idx],
                label=LABEL_NAMES[cls_idx] if ax_idx == 0 else "_nolegend_",
            )
            if pct > 4:
                ax.text(
                    left + pct / 2,
                    0,
                    f"{pct:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
            left += pct

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("% sequences", fontsize=7, color="#555554")
        ax.set_title(
            f"Bò #{int(cow_id)}", fontsize=9, fontweight="bold", color="#2C2C2A", pad=4
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CCCBC4")
        ax.tick_params(axis="x", colors="#555554", labelsize=7)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

        # Số lượng sample
        ax.text(
            100,
            0.4,
            f"n={total:,}",
            ha="right",
            va="top",
            fontsize=7,
            color="#888780",
        )

    # Ẩn subplot thừa
    for ax_idx in range(n_cows, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # Legend chung phía dưới
    legend_patches = [
        Patch(color=BEHAVIOR_COLORS[c], label=LABEL_NAMES[c])
        for c in range(num_classes)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=num_classes,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    out = save_dir / "01_label_dist_per_cow.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Phân phối nhãn per cow    → {out}")


#  def plot_label_distribution_per_cow(
#     cow_ids: torch.Tensor,
#     labels: torch.Tensor,
#     num_classes: int,
#     save_dir: Path,
# ):
#     """
#     Bar chart phân phối hành vi theo từng con bò.

#     Trục X: cow_id
#     Trục Y: số lượng sequences
#     Mỗi nhóm cột = 1 con bò
#     Mỗi màu = 1 behavior
#     """
#     unique_cows = sorted(cow_ids.unique().tolist())

#     # counts_matrix shape: (num_cows, num_classes)
#     counts_matrix = []
#     for cow_id in unique_cows:
#         mask = cow_ids == cow_id
#         cow_labels = labels[mask].tolist()
#         counts = [cow_labels.count(c) for c in range(num_classes)]
#         counts_matrix.append(counts)

#     counts_matrix = np.array(counts_matrix)

#     x = np.arange(len(unique_cows))
#     width = 0.11 if num_classes >= 7 else 0.8 / num_classes

#     fig, ax = plt.subplots(figsize=(13, 6), facecolor="white")

#     for cls_idx in range(num_classes):
#         offset = (cls_idx - (num_classes - 1) / 2) * width
#         bars = ax.bar(
#             x + offset,
#             counts_matrix[:, cls_idx],
#             width=width,
#             color=BEHAVIOR_COLORS[cls_idx],
#             label=LABEL_NAMES[cls_idx],
#             edgecolor="white",
#             linewidth=0.5,
#             zorder=3,
#         )

#         # Ghi số lên cột nếu đủ lớn
#         for bar in bars:
#             h = bar.get_height()
#             if h > 0:
#                 ax.text(
#                     bar.get_x() + bar.get_width() / 2,
#                     h + max(counts_matrix.max() * 0.01, 1),
#                     f"{int(h)}",
#                     ha="center",
#                     va="bottom",
#                     fontsize=7,
#                     color="#333333",
#                     rotation=90 if h < counts_matrix.max() * 0.08 else 0,
#                 )

#     ax.set_xticks(x)
#     ax.set_xticklabels([f"Cow {int(c)}" for c in unique_cows], fontsize=9)
#     ax.set_ylabel("Number of sequences", fontsize=10, color="#555554")
#     ax.set_xlabel("Cow ID", fontsize=10, color="#555554")

#     ax.set_title(
#         "Behavior Distribution per Cow",
#         fontsize=14,
#         fontweight="bold",
#         color="#2C2C2A",
#         pad=14,
#     )

#     ax.set_facecolor("#FAFAF8")
#     ax.spines[["top", "right"]].set_visible(False)
#     ax.spines[["left", "bottom"]].set_color("#CCCBC4")
#     ax.tick_params(colors="#555554", labelsize=9)
#     ax.grid(axis="y", color="#E0DFD8", linewidth=0.7, zorder=0)

#     ax.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, -0.14),
#         ncol=num_classes,
#         fontsize=8,
#         frameon=False,
#     )

#     plt.tight_layout()
#     out = save_dir / "01_label_dist_per_cow_bar.png"
#     plt.savefig(out, dpi=180, bbox_inches="tight")
#     plt.close()

#     print(f"[Plot]  Label distribution per cow  → {out}")


# ── 2. Phân phối split (train / val / test) mỗi fold ─────────────────────────


def plot_split_distribution(
    full_dataset: Dataset,
    folds: list[dict],
    num_classes: int,
    save_dir: Path,
):
    """
    Vẽ phân phối nhãn theo Train / Val / Test cho từng fold.

    Mỗi subplot = 1 fold.
    Trong mỗi subplot:
        X-axis = Train, Val, Test
        Y-axis = % sequences
        Bar được stacked theo behavior.
    """

    # Tương thích với cả full_dataset trực tiếp và wrapper cũ
    if hasattr(full_dataset, "labels"):
        labels_all = full_dataset.labels
    elif hasattr(full_dataset, "dataset") and hasattr(full_dataset.dataset, "labels"):
        labels_all = full_dataset.dataset.labels
    else:
        raise ValueError(
            "Không tìm thấy labels trong full_dataset. "
            "Cần truyền CowDataset hoặc wrapper có .dataset.labels."
        )

    split_names = ["Train", "Val", "Test"]
    split_keys = ["train_idx", "val_idx", "test_idx"]
    cow_keys = ["train_cows", "val_cows", "test_cows"]

    n_folds = len(folds)
    ncols = 3
    nrows = int(np.ceil(n_folds / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 4.6 * nrows),
        facecolor="white",
    )

    axes = np.array(axes).reshape(-1)

    fig.suptitle(
        "Behavior Distribution by Split — 5-Fold Object-wise Cross-validation",
        fontsize=15,
        fontweight="bold",
        color="#2C2C2A",
        y=1.02,
    )

    for fold_idx, fold in enumerate(folds):
        ax = axes[fold_idx]

        # pct_matrix shape: (3 splits, num_classes)
        pct_matrix = []
        count_matrix = []
        split_totals = []

        for skey in split_keys:
            idx = fold[skey]
            split_labels = labels_all[idx].tolist()
            total = len(split_labels)
            split_totals.append(total)

            counts = np.array(
                [split_labels.count(c) for c in range(num_classes)],
                dtype=float,
            )
            count_matrix.append(counts)

            if total > 0:
                pct_matrix.append(counts / total * 100)
            else:
                pct_matrix.append(np.zeros(num_classes))

        pct_matrix = np.array(pct_matrix)
        count_matrix = np.array(count_matrix)

        x = np.arange(len(split_names))
        bottom = np.zeros(len(split_names))

        for cls_idx in range(num_classes):
            values = pct_matrix[:, cls_idx]

            bars = ax.bar(
                x,
                values,
                bottom=bottom,
                color=BEHAVIOR_COLORS[cls_idx],
                label=LABEL_NAMES[cls_idx] if fold_idx == 0 else "_nolegend_",
                edgecolor="white",
                linewidth=0.7,
                width=0.58,
                zorder=3,
            )

            # Ghi % lên segment nếu đủ lớn
            for i, bar in enumerate(bars):
                h = bar.get_height()
                if h >= 4:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom[i] + h / 2,
                        f"{h:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )

            bottom += values

        # Ghi số lượng sample dưới nhãn split
        xtick_labels = []
        for sname, total, ckey in zip(split_names, split_totals, cow_keys):
            cows = fold[ckey]
            xtick_labels.append(f"{sname}\n" f"n={total:,}\n" f"cows={cows}")

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% sequences", fontsize=9, color="#555554")

        ax.set_title(
            f"Fold {fold['fold']}",
            fontsize=12,
            fontweight="bold",
            color="#2C2C2A",
            pad=10,
        )

        ax.set_facecolor("#FAFAF8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CCCBC4")
        ax.tick_params(colors="#555554", labelsize=8)
        ax.grid(axis="y", color="#E0DFD8", linewidth=0.7, zorder=0)

    # Ẩn subplot thừa nếu có
    for ax_idx in range(n_folds, len(axes)):
        axes[ax_idx].set_visible(False)

    handles = [
        Patch(color=BEHAVIOR_COLORS[c], label=LABEL_NAMES[c])
        for c in range(num_classes)
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=num_classes,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    out = save_dir / "02_split_distribution_bar.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[Plot]  Split distribution bar      → {out}")


# ── 3. Loss curves (1 figure, 5 fold × 2 subplots) ───────────────────────────


def plot_all_loss_curves(
    all_histories: list[dict],
    save_dir: Path,
):
    nrows, ncols = len(all_histories), 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 5, nrows * 2.8),
        facecolor="white",
    )
    fig.suptitle(
        "Loss & Accuracy curves — 5 fold",
        fontsize=12,
        fontweight="bold",
        color="#2C2C2A",
    )

    for row, hist in enumerate(all_histories):
        fold_n = hist["fold"]
        ep = range(1, len(hist["train_loss"]) + 1)
        best = hist["best_epoch"]

        # Loss
        ax = axes[row][0]
        ax.plot(ep, hist["train_loss"], color="#378ADD", lw=1.5, label="Train")
        ax.plot(ep, hist["val_loss"], color="#D85A30", lw=1.5, label="Val")
        ax.axvline(
            best, color="#1D9E75", lw=1, ls="--", alpha=0.8, label=f"Best ep {best}"
        )
        _style_ax(ax, title=f"Fold {fold_n} — Loss", xlabel="Epoch", ylabel="Loss")
        ax.legend(fontsize=7, frameon=False)

        # Accuracy
        ax = axes[row][1]
        ax.plot(ep, hist["train_acc"], color="#378ADD", lw=1.5, label="Train")
        ax.plot(ep, hist["val_acc"], color="#D85A30", lw=1.5, label="Val")
        ax.axvline(
            best, color="#1D9E75", lw=1, ls="--", alpha=0.8, label=f"Best ep {best}"
        )
        _style_ax(
            ax, title=f"Fold {fold_n} — Accuracy", xlabel="Epoch", ylabel="Accuracy"
        )
        ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    out = save_dir / "03_loss_curves_all_folds.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Loss curves all folds      → {out}")


# ── 4. Per-fold confusion matrices (5 subplots) ───────────────────────────────


def plot_per_fold_confusion_matrices(
    fold_cms: list[
        np.ndarray
    ],  # list 5 phần tử, mỗi phần tử (num_classes, num_classes)
    num_classes: int,
    save_dir: Path,
):
    """
    5 confusion matrix riêng lẻ (normalized recall) trong 1 figure.
    Hữu ích để kiểm tra xem fold nào mô hình yếu hơn.
    """
    ncols = 5
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(ncols * 3.6, 4.2),
        facecolor="white",
    )
    fig.suptitle(
        "Confusion matrix từng fold (normalized — recall per class)",
        fontsize=11,
        fontweight="bold",
        color="#2C2C2A",
    )

    short = ["Wlk", "Std", "FdU", "FdD", "Lck", "Drk", "Lie"][:num_classes]

    for col, (ax, cm_raw) in enumerate(zip(axes, fold_cms)):
        row_sums = cm_raw.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm_raw / row_sums

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(
            f"Fold {col+1}", fontsize=9, fontweight="bold", color="#2C2C2A", pad=4
        )
        if col == 0:
            ax.set_ylabel("True", fontsize=7, color="#555554")
        ax.set_xlabel("Pred", fontsize=7, color="#555554")

        thresh = 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                val = cm_norm[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if val > thresh else "#333",
                )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = save_dir / "04_confusion_matrix_per_fold.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Per-fold confusion matrices → {out}")


# ── 5. Merged confusion matrix (gom 5 fold) ──────────────────────────────────


def plot_merged_confusion_matrix(
    cm_merged: np.ndarray,
    num_classes: int,
    save_dir: Path,
):
    """
    1 confusion matrix duy nhất tổng hợp từ tất cả 5 fold.
    Hiện cả count và normalized (recall) side-by-side.
    """
    row_sums = cm_merged.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_merged / row_sums

    label_names = LABEL_NAMES[:num_classes]
    short = ["Wlk", "Std", "FdU", "FdD", "Lck", "Drk", "Lie"][:num_classes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle(
        "Merged confusion matrix — tổng hợp 5 fold (toàn bộ test samples)",
        fontsize=12,
        fontweight="bold",
        color="#2C2C2A",
    )

    for ax, data, title, fmt, cmap in zip(
        axes,
        [cm_merged, cm_norm],
        ["Count (raw)", "Normalized (recall per class)"],
        ["d", ".2f"],
        ["YlOrRd", "Blues"],
    ):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10, fontweight="bold", color="#2C2C2A", pad=6)
        ax.set_xlabel("Predicted", fontsize=9, color="#555554")
        ax.set_ylabel("True", fontsize=9, color="#555554")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(label_names, fontsize=8)

        thresh = data.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                val = data[i, j]
                text = f"{int(val)}" if fmt == "d" else f"{val:.2f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if val > thresh else "#333",
                )

    plt.tight_layout()
    out = save_dir / "05_confusion_matrix_merged.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Merged confusion matrix    → {out}")


# ── 6. Per-class F1 với mean ± std qua 5 fold ────────────────────────────────


def plot_per_class_f1(
    fold_f1s: list[list[float]],  # shape (5, num_classes)
    num_classes: int,
    save_dir: Path,
):
    """
    Bar chart F1 per class.
    Mỗi bar = mean qua 5 fold, error bar = std.
    Đường ngang = macro mean.
    """
    arr = np.array(fold_f1s)  # (5, num_classes)
    mean = arr.mean(axis=0)  # (num_classes,)
    std = arr.std(axis=0)  # (num_classes,)
    macro_mean = mean.mean()

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    x = np.arange(num_classes)

    bars = ax.bar(
        x,
        mean,
        yerr=std,
        capsize=4,
        color=BEHAVIOR_COLORS[:num_classes],
        width=0.55,
        error_kw={"elinewidth": 1.2, "ecolor": "#888780"},
        zorder=3,
    )

    # Giá trị mean trên mỗi bar
    for bar, m, s in zip(bars, mean, std):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.015,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333",
        )

    # Đường macro mean
    ax.axhline(
        macro_mean,
        color="#D85A30",
        lw=1.5,
        ls="--",
        label=f"Macro mean = {macro_mean:.3f}",
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES[:num_classes], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, min(1.15, mean.max() + std.max() + 0.15))
    ax.set_ylabel("F1 (macro)", fontsize=9, color="#555554")
    _style_ax(ax, title="Per-class F1 — mean ± std qua 5 fold")
    ax.legend(fontsize=8.5, frameon=False)

    # Bảng nhỏ mean ± std
    col_labels = LABEL_NAMES[:num_classes]
    table_vals = [[f"{m:.3f}±{s:.3f}" for m, s in zip(mean, std)]]
    tbl = ax.table(
        cellText=table_vals,
        colLabels=col_labels,
        loc="bottom",
        cellLoc="center",
        bbox=[0, -0.38, 1, 0.22],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCBC4")
        cell.set_facecolor("#FAFAF8" if row == 0 else "white")
        if row == 0:
            cell.set_text_props(fontweight="bold", color="#2C2C2A")

    plt.tight_layout()
    out = save_dir / "06_per_class_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Per-class F1 mean±std      → {out}")


# ── 7. Summary dashboard ──────────────────────────────────────────────────────


def plot_summary_dashboard(
    fold_results: list[dict],  # list of {fold, test_f1_macro, test_acc, ...}
    save_dir: Path,
):
    """
    1 figure tổng hợp: F1 macro mỗi fold + bảng kết quả.
    """
    folds_n = [r["fold"] for r in fold_results]
    f1s = [r["test_f1_macro"] for r in fold_results]
    accs = [r["test_acc"] for r in fold_results]
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    mean_acc = np.mean(accs)

    fig = plt.figure(figsize=(10, 4.5), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    fig.suptitle(
        "Kết quả 5-fold cross-validation — Object-wise Split",
        fontsize=12,
        fontweight="bold",
        color="#2C2C2A",
    )

    # Bar chart F1 macro per fold
    bar_colors = ["#378ADD"] * len(folds_n)
    bars = ax1.bar(
        [f"Fold {n}" for n in folds_n],
        f1s,
        color=bar_colors,
        width=0.5,
        zorder=3,
    )
    for bar, f1 in zip(bars, f1s):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            f1 + 0.005,
            f"{f1:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax1.axhline(
        mean_f1,
        color="#D85A30",
        lw=1.5,
        ls="--",
        label=f"Mean = {mean_f1:.3f} ± {std_f1:.3f}",
        zorder=4,
    )
    ax1.set_ylim(0, min(1.0, max(f1s) + 0.12))
    ax1.set_ylabel("F1 macro", fontsize=9, color="#555554")
    _style_ax(ax1, title="F1 macro per fold")
    ax1.legend(fontsize=8.5, frameon=False)

    # Bảng tóm tắt
    ax2.axis("off")
    summary_data = [
        ["Metric", "Mean", "Std"],
        ["F1 macro", f"{mean_f1:.4f}", f"± {std_f1:.4f}"],
        ["Accuracy", f"{mean_acc:.4f}", ""],
        ["", "", ""],
    ] + [
        [
            f"Fold {r['fold']}",
            f"F1={r['test_f1_macro']:.3f}",
            f"Acc={r['test_acc']:.3f}",
        ]
        for r in fold_results
    ]
    tbl = ax2.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCBC4")
        if row == 0:
            cell.set_facecolor("#E6F1FB")
            cell.set_text_props(fontweight="bold", color="#0C447C")
        elif row in (1, 2):
            cell.set_facecolor("#EAF3DE")
        else:
            cell.set_facecolor("white")

    plt.tight_layout()
    out = save_dir / "07_summary_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Summary dashboard          → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CowBehaviorModel — 5-fold Object-wise Cross-validation")
    print(f"{'='*60}")
    print(f"Device    : {device}")
    print(f"Data      : {args.data}")
    print(f"Output    : {args.output}")
    print(f"Metric    : F1 macro")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[Data]  Loading {args.data} ...")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    sensor = data["sensor"]  # (B, T, 256)
    image = data["image"]  # (B, T, 256, 8, 8)
    labels = data["labels"]  # (B,)
    cow_ids = data["cow_ids"]  # (B,)
    T = data.get("T", sensor.shape[1])

    print(f"[Data]  sensor  : {tuple(sensor.shape)}")
    print(f"[Data]  image   : {tuple(image.shape)}")
    print(
        f"[Data]  labels  : {tuple(labels.shape)}  classes={labels.unique().tolist()}"
    )
    print(
        f"[Data]  cows    : {cow_ids.unique().numel()} con — {sorted(cow_ids.unique().tolist())}"
    )

    full_dataset = CowDataset(sensor, image, labels, cow_ids, T=T)

    # ── EDA: phân phối nhãn từng con bò ──────────────────────────────────────
    plot_label_distribution_per_cow(cow_ids, labels, args.num_classes, save_dir)

    # ── Build 5 folds ─────────────────────────────────────────────────────────
    folds = build_folds(cow_ids, n_folds=5)

    # ── Vẽ split distribution (cần full_dataset wrapper) ─────────────────────
    # Wrap tạm để dùng chung interface
    class _DatasetWrapper:
        def __init__(self, ds):
            self.dataset = ds

    plot_split_distribution(
        _DatasetWrapper(full_dataset), folds, args.num_classes, save_dir
    )

    # ── Containers tích lũy qua 5 fold ────────────────────────────────────────
    all_true: list[int] = []
    all_pred: list[int] = []
    fold_cms: list[np.ndarray] = []
    fold_f1s: list[list[float]] = []  # (5, num_classes)
    fold_results: list[dict] = []
    all_histories: list[dict] = []

    # ── 5-fold loop ───────────────────────────────────────────────────────────
    for fold in folds:
        fold_n = fold["fold"]
        print(f"\n{'='*60}")
        print(f"FOLD {fold_n}/5")
        print(f"  Train bò: {fold['train_cows']}  ({len(fold['train_idx'])} seq)")
        print(f"  Val   bò: {fold['val_cows']}    ({len(fold['val_idx'])} seq)")
        print(f"  Test  bò: {fold['test_cows']}   ({len(fold['test_idx'])} seq)")
        print(f"{'='*60}")

        train_set = Subset(full_dataset, fold["train_idx"])
        val_set = Subset(full_dataset, fold["val_idx"])
        test_set = Subset(full_dataset, fold["test_idx"])

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # Class weights chỉ từ train fold này
        train_labels = labels[fold["train_idx"]]
        class_weights = compute_class_weights(
            train_labels, num_classes=args.num_classes, device=device
        )

        # Model mới cho mỗi fold (không dùng lại weights)
        model = CowBehaviorModel(
            num_classes=args.num_classes,
            d_model=256,
            nhead=8,
            num_encoder_layers=2,
            dim_feedforward=1024,
            dropout=args.dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        history = {
            "fold": fold_n,
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "best_epoch": 1,
        }
        best_val_f1 = -1.0
        patience_counter = 0
        ckpt_path = save_dir / f"best_model_fold{fold_n}.pt"

        print(
            f"\n{'Epoch':>6} | {'TrLoss':>8} | {'TrAcc':>7} | "
            f"{'VLoss':>8} | {'VAcc':>7} | {'VF1':>7} | Time"
        )
        print("-" * 65)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            vl_loss, vl_acc, vl_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step(vl_loss)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)
            history["val_f1"].append(vl_f1)

            marker = " ◀" if vl_f1 > best_val_f1 else ""
            print(
                f"{epoch:>6} | {tr_loss:>8.4f} | {tr_acc:>7.4f} | "
                f"{vl_loss:>8.4f} | {vl_acc:>7.4f} | {vl_f1:>7.4f} | "
                f"{time.time()-t0:.1f}s{marker}"
            )

            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                patience_counter = 0
                history["best_epoch"] = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_f1": vl_f1,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[Early Stop] Fold {fold_n} dừng tại epoch {epoch}.")
                    break

        all_histories.append(history)

        # ── Test fold này ──────────────────────────────────────────────────────
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        te_loss, te_acc, te_f1, te_preds, te_true = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"\n[Fold {fold_n} Test]  Loss={te_loss:.4f}  "
            f"Acc={te_acc:.4f}  F1_macro={te_f1:.4f}"
        )
        print(
            classification_report(
                te_true,
                te_preds,
                labels=list(range(args.num_classes)),
                target_names=LABEL_NAMES[: args.num_classes],
                zero_division=0,
            )
        )
        # Tích lũy predictions
        all_true.extend(te_true)
        all_pred.extend(te_preds)

        # Confusion matrix riêng fold này
        cm_fold = confusion_matrix(
            te_true, te_preds, labels=list(range(args.num_classes))
        )
        fold_cms.append(cm_fold)

        # F1 per class fold này
        from sklearn.metrics import f1_score as f1_per

        f1_per_class = f1_per(
            te_true,
            te_preds,
            average=None,
            labels=list(range(args.num_classes)),
            zero_division=0,
        ).tolist()
        fold_f1s.append(f1_per_class)

        fold_results.append(
            {
                "fold": fold_n,
                "test_f1_macro": te_f1,
                "test_acc": te_acc,
                "test_loss": te_loss,
                "best_epoch": history["best_epoch"],
            }
        )

    # ── Tổng hợp sau 5 fold ───────────────────────────────────────────────────
    cm_merged = np.sum(fold_cms, axis=0)

    macro_f1s = [r["test_f1_macro"] for r in fold_results]
    print(f"\n{'='*60}")
    print(f"5-FOLD SUMMARY")
    print(f"  F1 macro per fold : {[f'{f:.4f}' for f in macro_f1s]}")
    print(f"  Mean F1 macro     : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(
        f"  Overall (merged)  : "
        f"F1={f1_score(all_true, all_pred, average='macro', zero_division=0):.4f}  "
        f"Acc={accuracy_score(all_true, all_pred):.4f}"
    )
    print(f"{'='*60}")

    # ── Vẽ tất cả plots ───────────────────────────────────────────────────────
    print("\n[Plot]  Đang vẽ tất cả figures...")
    plot_all_loss_curves(all_histories, save_dir)
    plot_per_fold_confusion_matrices(fold_cms, args.num_classes, save_dir)
    plot_merged_confusion_matrix(cm_merged, args.num_classes, save_dir)
    plot_per_class_f1(fold_f1s, args.num_classes, save_dir)
    plot_summary_dashboard(fold_results, save_dir)

    # ── Lưu text summary ──────────────────────────────────────────────────────
    results_path = save_dir / "results_summary.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("CowBehaviorModel — 5-fold Object-wise Cross-validation\n")
        f.write(f"{'='*60}\n")
        f.write(f"Metric chính: F1 MACRO\n\n")
        for r in fold_results:
            f.write(
                f"Fold {r['fold']}: F1={r['test_f1_macro']:.4f}  "
                f"Acc={r['test_acc']:.4f}  best_ep={r['best_epoch']}\n"
            )
        f.write(
            f"\nMean F1 macro : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}\n"
        )
        f.write(f"\nMerged classification report:\n")
        f.write(
            classification_report(
                all_true,
                all_pred,
                labels=list(range(args.num_classes)),
                target_names=LABEL_NAMES[: args.num_classes],
                zero_division=0,
            )
        )
    print(f"[Save]  Results summary → {results_path}")
    print(f"[Done]  Tất cả output → {save_dir}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = SimpleNamespace(
        data=r"D:\nhung kien thuc dai hoc\Semester 05\DPL\mmcow_git\fusion_dataset_T16.pt",
        output=r"D:\nhung kien thuc dai hoc\Semester 05\DPL\mmcow_git\runs\exp_cv",
        # Model
        num_classes=7,
        dropout=0.2,
        # Training
        epochs=50,
        batch_size=32,
        lr=1e-4,
        patience=5,
        num_workers=0,
        seed=42,
    )
    train(args)
