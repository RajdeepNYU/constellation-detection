"""Build the hero figure for the README.

Reads the three labelled training scenes, a train-split prediction CSV and the
fitted constellation models from its diagnostics file (written next to the CSV by
src/main.py), and draws the recovered figure on each sky image.

usage:
    python src/main.py --data participant --split train --balanced --output runs/train_predictions.csv
    python tools/make_hero_figure.py participant runs/train_predictions.csv assets/hero.png
"""
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(sys.argv[1])
PREDICTIONS = Path(sys.argv[2])
OUT = Path(sys.argv[3])


def raw_nodes(path):
    """Same node extraction as patterns.read_stars, without normalisation."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    mask = np.all(image[:, :, :3] > 180, axis=2)
    if image.shape[2] == 4:
        mask &= image[:, :, 3] > 100
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    keep = stats[1:, cv2.CC_STAT_AREA] >= 8
    return centers[1:][keep], image


def pattern_edges(path):
    nodes, image = raw_nodes(path)
    ink = image[:, :, 3] > 60 if image.shape[2] == 4 else image[:, :, :3].max(2) > 60
    edges = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            length = np.linalg.norm(b - a)
            if length < 6:
                continue
            # Skip segments that pass through another node (would duplicate a chain).
            blocked = False
            for k in range(n):
                if k in (i, j):
                    continue
                t = np.dot(nodes[k] - a, b - a) / (length**2)
                if 0.05 < t < 0.95:
                    d = np.linalg.norm(a + t * (b - a) - nodes[k])
                    if d < 6:
                        blocked = True
                        break
            if blocked:
                continue
            ts = np.linspace(0.12, 0.88, max(int(length), 10))
            pts = a[None] + ts[:, None] * (b - a)[None]
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, ink.shape[1] - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, ink.shape[0] - 1)
            if ink[ys, xs].mean() > 0.92:
                edges.append((i, j))
    return edges


def parse_cell(cell):
    cell = cell.strip()
    if cell in ("-1", ""):
        return None
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", cell)]
    return nums[0], nums[1], int(nums[2]) if len(nums) > 2 else 0


def read_rows(path):
    with open(path, newline="") as f:
        return {row["Id"]: row for row in csv.DictReader(f)}


preds = read_rows(PREDICTIONS)
truth = read_rows(DATA / "train_ground_truth.csv")
diag = json.loads(PREDICTIONS.with_suffix(".diagnostics.json").read_text())

scenes = ["pisces", "scorpius", "taurus"]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), facecolor="#05070d")

for ax, scene in zip(axes, scenes):
    image = cv2.imread(str(DATA / "train" / scene / f"{scene}_image.png"), cv2.IMREAD_GRAYSCALE)
    row = preds[scene]
    name = row["constellation"]
    model = next(m for m in diag[scene]["models"] if m["name"] == name)
    projected = np.array(model["projected"], dtype=float)
    edges = pattern_edges(DATA / "patterns" / f"{name}_pattern.png")

    n = int(row["n_patches"])
    figure_pts, other_pts = [], []
    for k in range(1, n + 1):
        cell = parse_cell(row[f"patch_{k:02d}"])
        if cell is None:
            continue
        (figure_pts if cell[2] == 1 else other_pts).append(cell[:2])
    figure_pts, other_pts = np.array(figure_pts), np.array(other_pts)

    # Crop to the recovered figure (projected nodes + figure queries), clipped to the image.
    pts = np.vstack([projected, figure_pts]) if len(figure_pts) else projected
    inside = pts[(pts[:, 0] >= 0) & (pts[:, 0] < 3000) & (pts[:, 1] >= 0) & (pts[:, 1] < 3000)]
    x0, y0 = inside.min(0) - 260
    x1, y1 = inside.max(0) + 260
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = int(max(0, cx - side / 2)), int(min(3000, cx + side / 2))
    y0, y1 = int(max(0, cy - side / 2)), int(min(3000, cy + side / 2))
    crop = image[y0:y1, x0:x1].astype(np.float32)
    lo, hi = np.percentile(crop, [1, 99.7])
    crop = np.clip((crop - lo) / max(hi - lo, 1), 0, 1) ** 0.8

    ax.imshow(crop, cmap="bone", extent=(x0, x1, y1, y0), interpolation="lanczos")
    for i, j in edges:
        (ax1, ay1), (ax2, ay2) = projected[i], projected[j]
        ax.plot([ax1, ax2], [ay1, ay2], color="#7fd3ff", lw=1.6, alpha=0.75, zorder=2)
    ax.scatter(projected[:, 0], projected[:, 1], s=28, facecolors="none",
               edgecolors="#7fd3ff", linewidths=1.0, alpha=0.8, zorder=3)
    if len(other_pts):
        ax.scatter(other_pts[:, 0], other_pts[:, 1], s=55, marker="s", facecolors="none",
                   edgecolors="#c7c9d1", linewidths=1.0, alpha=0.7, zorder=3)
    if len(figure_pts):
        ax.scatter(figure_pts[:, 0], figure_pts[:, 1], s=170, facecolors="none",
                   edgecolors="#ffc94a", linewidths=2.2, zorder=4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_axis_off()
    correct = "correct" if truth[scene]["constellation"] == name else "wrong"
    ax.set_title(f"{name.replace('-', ' ').title()}", color="white", fontsize=19,
                 fontweight="bold", pad=10)
    ax.text(0.5, -0.045, f"identified from 48 candidates ({correct})  ·  {len(figure_pts)} queries assigned to the figure",
            transform=ax.transAxes, ha="center", va="top", color="#aab3c5", fontsize=11)

handles = [
    plt.Line2D([], [], color="#7fd3ff", lw=1.6, label="reference figure, projected by the fitted homography"),
    plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor="none",
               markeredgecolor="#ffc94a", markeredgewidth=2.2, label="query patch localised as a figure star"),
    plt.Line2D([], [], marker="s", ls="", markersize=8, markerfacecolor="none",
               markeredgecolor="#c7c9d1", label="query patch localised off-figure"),
]
leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11.5,
                 bbox_to_anchor=(0.5, 0.0))
for text in leg.get_texts():
    text.set_color("#d7dce6")
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.12, wspace=0.04)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=110, facecolor=fig.get_facecolor())
print("wrote", OUT)
for scene in scenes:
    print(scene, preds[scene]["constellation"], "truth:", truth[scene]["constellation"])
