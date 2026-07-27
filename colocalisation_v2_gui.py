#!/usr/bin/env python3
"""
Comprehensive Colocalization Analysis Tool — GUI Edition
=========================================================
Analyzes spatial colocalization between two populations from CSV files.

Features:
  - Tkinter GUI for file/parameter selection
  - Timestamped output files (never overwritten)
  - 9 analysis modules with publication-quality figures
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from scipy.spatial import (
    distance_matrix, KDTree, Voronoi, voronoi_plot_2d, ConvexHull
)
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from scipy.ndimage import gaussian_filter

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import networkx as nx

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
COLOR_A = "#E63946"
COLOR_B = "#457B9D"
COLOR_BOTH = "#2A9D8F"
FIGSIZE_LARGE = (14, 12)
FIGSIZE_WIDE = (16, 7)
FIGSIZE_SQ = (10, 10)
DPI = 150


# ──────────────────────────────────────────────────────────────────────────────
# TIMESTAMP UTILITY
# ──────────────────────────────────────────────────────────────────────────────
def generate_run_id():
    """Generate a unique run ID from date + time."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stamped_filename(base_name: str, run_id: str, ext: str = ".png") -> str:
    """Return filename with run_id appended before extension."""
    return f"{base_name}_{run_id}{ext}"


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and keep relevant columns."""
    df = pd.read_csv(path, sep=None, engine="python")
    needed = {"centroid_x", "centroid_y", "area"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[["centroid_x", "centroid_y", "area"]].copy()


def compute_extent(df_a, df_b, pad_frac=0.05):
    xs = np.concatenate([df_a["centroid_x"].values, df_b["centroid_x"].values])
    ys = np.concatenate([df_a["centroid_y"].values, df_b["centroid_y"].values])
    pad_x = (xs.max() - xs.min()) * pad_frac
    pad_y = (ys.max() - ys.min()) * pad_frac
    return (xs.min() - pad_x, xs.max() + pad_x,
            ys.min() - pad_y, ys.max() + pad_y)


def add_scale_bar(ax, length, label=None, loc="lower right", fontsize=9):
    if label is None:
        label = f"{length:.0f} units"
    bar = AnchoredSizeBar(
        ax.transData, length, label, loc,
        pad=0.5, color="black", frameon=False,
        size_vertical=length * 0.02,
        fontproperties=fm.FontProperties(size=fontsize),
    )
    ax.add_artist(bar)


def area_to_marker_size(areas, min_s=15, max_s=200):
    if areas.max() == areas.min():
        return np.full(len(areas), (min_s + max_s) / 2)
    norm = (areas - areas.min()) / (areas.max() - areas.min())
    return min_s + norm * (max_s - min_s)


def coords(df):
    return df[["centroid_x", "centroid_y"]].values


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SPATIAL OVERLAY
# ──────────────────────────────────────────────────────────────────────────────
def plot_spatial_overlay(df_a, df_b, name_a, name_b, extent, outdir, run_id):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=DPI)
    sa = area_to_marker_size(df_a["area"].values)
    sb = area_to_marker_size(df_b["area"].values)

    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        add_scale_bar(ax, (extent[1] - extent[0]) * 0.15)

    axes[0].scatter(df_a["centroid_x"], df_a["centroid_y"], s=sa,
                    c=COLOR_A, alpha=0.6, edgecolors="k", linewidths=0.3)
    axes[0].set_title(f"{name_a}  (n={len(df_a)})", fontsize=13)

    axes[1].scatter(df_b["centroid_x"], df_b["centroid_y"], s=sb,
                    c=COLOR_B, alpha=0.6, edgecolors="k", linewidths=0.3)
    axes[1].set_title(f"{name_b}  (n={len(df_b)})", fontsize=13)

    axes[2].scatter(df_a["centroid_x"], df_a["centroid_y"], s=sa,
                    c=COLOR_A, alpha=0.5, edgecolors="k", linewidths=0.2,
                    label=name_a)
    axes[2].scatter(df_b["centroid_x"], df_b["centroid_y"], s=sb,
                    c=COLOR_B, alpha=0.5, edgecolors="k", linewidths=0.2,
                    label=name_b)
    axes[2].legend(fontsize=11, loc="upper right")
    axes[2].set_title("Overlay", fontsize=13)

    for ax in axes:
        ax.set_xlabel("X"); ax.set_ylabel("Y")

    fig.suptitle("Spatial Distribution Overview", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("01_spatial_overlay", run_id),
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Spatial overlay saved.")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  NEAREST-NEIGHBOUR DISTANCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def nearest_neighbour_analysis(df_a, df_b, name_a, name_b, extent, outdir,
                               run_id):
    ca, cb = coords(df_a), coords(df_b)
    tree_a, tree_b = KDTree(ca), KDTree(cb)
    dist_ab, idx_ab = tree_b.query(ca)
    dist_ba, idx_ba = tree_a.query(cb)

    area_roi = (extent[1] - extent[0]) * (extent[3] - extent[2])
    lambda_b = len(df_b) / area_roi
    lambda_a = len(df_a) / area_roi
    expected_ab = 0.5 / np.sqrt(lambda_b) if lambda_b > 0 else np.nan
    expected_ba = 0.5 / np.sqrt(lambda_a) if lambda_a > 0 else np.nan

    stats = {
        f"{name_a}→{name_b} mean NN dist": np.mean(dist_ab),
        f"{name_a}→{name_b} median NN dist": np.median(dist_ab),
        f"{name_b}→{name_a} mean NN dist": np.mean(dist_ba),
        f"{name_b}→{name_a} median NN dist": np.median(dist_ba),
        f"Expected NN dist (CSR) {name_a}→{name_b}": expected_ab,
        f"Expected NN dist (CSR) {name_b}→{name_a}": expected_ba,
        f"Coloc. Index {name_a}→{name_b} (obs/exp)":
            np.mean(dist_ab) / expected_ab if expected_ab else np.nan,
        f"Coloc. Index {name_b}→{name_a} (obs/exp)":
            np.mean(dist_ba) / expected_ba if expected_ba else np.nan,
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=DPI)
    bins = np.linspace(0, max(dist_ab.max(), dist_ba.max()), 50)
    axes[0].hist(dist_ab, bins=bins, color=COLOR_A, alpha=0.65,
                 label=f"{name_a}→{name_b}", edgecolor="k", linewidth=0.3)
    axes[0].hist(dist_ba, bins=bins, color=COLOR_B, alpha=0.65,
                 label=f"{name_b}→{name_a}", edgecolor="k", linewidth=0.3)
    axes[0].axvline(expected_ab, ls="--", color=COLOR_A, lw=1.5,
                    label=f"CSR expect. {name_a}→{name_b}")
    axes[0].axvline(expected_ba, ls="--", color=COLOR_B, lw=1.5,
                    label=f"CSR expect. {name_b}→{name_a}")
    axes[0].set_xlabel("NN Distance"); axes[0].set_ylabel("Count")
    axes[0].set_title("NN Distance Distributions"); axes[0].legend(fontsize=8)

    for d, c, lab in [(dist_ab, COLOR_A, f"{name_a}→{name_b}"),
                       (dist_ba, COLOR_B, f"{name_b}→{name_a}")]:
        sorted_d = np.sort(d)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        axes[1].plot(sorted_d, cdf, color=c, lw=2, label=lab)
    axes[1].set_xlabel("Distance"); axes[1].set_ylabel("Cumulative Fraction")
    axes[1].set_title("Cumulative NN Distance"); axes[1].legend(fontsize=9)

    ax = axes[2]
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    norm = Normalize(vmin=0, vmax=np.percentile(dist_ab, 95))
    segments = [[ca[i], cb[j]] for i, j in enumerate(idx_ab)]
    colors_seg = [dist_ab[i] for i in range(len(idx_ab))]
    lc = LineCollection(segments, cmap="viridis_r", norm=norm,
                        linewidths=0.6, alpha=0.5)
    lc.set_array(np.array(colors_seg))
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NN Distance")
    ax.scatter(ca[:, 0], ca[:, 1], s=12, c=COLOR_A, zorder=3, label=name_a)
    ax.scatter(cb[:, 0], cb[:, 1], s=12, c=COLOR_B, zorder=3, label=name_b)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"NN Links {name_a}→{name_b}")
    add_scale_bar(ax, (extent[1] - extent[0]) * 0.15)

    fig.suptitle("Nearest-Neighbour Distance Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("02_nearest_neighbour", run_id),
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Nearest-neighbour analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 3.  GRID-BASED INTENSITY CORRELATION
# ──────────────────────────────────────────────────────────────────────────────
def grid_correlation_analysis(df_a, df_b, name_a, name_b, extent, outdir,
                              run_id, n_bins=30):
    x_edges = np.linspace(extent[0], extent[1], n_bins + 1)
    y_edges = np.linspace(extent[2], extent[3], n_bins + 1)
    hist_a, _, _ = np.histogram2d(df_a["centroid_x"], df_a["centroid_y"],
                                   bins=[x_edges, y_edges])
    hist_b, _, _ = np.histogram2d(df_b["centroid_x"], df_b["centroid_y"],
                                   bins=[x_edges, y_edges])
    ha, hb = hist_a.ravel().astype(float), hist_b.ravel().astype(float)

    r_pearson, p_pearson = pearsonr(ha, hb)
    r_spearman, p_spearman = spearmanr(ha, hb)
    mask_b = hb > 0; mask_a = ha > 0
    M1 = ha[mask_b].sum() / ha.sum() if ha.sum() > 0 else 0
    M2 = hb[mask_a].sum() / hb.sum() if hb.sum() > 0 else 0

    n_rand = 200
    rand_r = np.zeros(n_rand)
    for i in range(n_rand):
        hb_shuf = hb.copy(); np.random.shuffle(hb_shuf)
        rand_r[i], _ = pearsonr(ha, hb_shuf)
    costes_p = np.mean(rand_r >= r_pearson)

    stats = {
        "Pearson r": r_pearson, "Pearson p-value": p_pearson,
        "Spearman ρ": r_spearman, "Spearman p-value": p_spearman,
        f"Manders M1 (frac {name_a} in {name_b})": M1,
        f"Manders M2 (frac {name_b} in {name_a})": M2,
        "Costes randomisation p": costes_p,
    }

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=DPI)
    vmax = max(hist_a.max(), hist_b.max())
    im0 = axes[0].imshow(hist_a.T, origin="lower", cmap="Reds",
                          extent=[extent[0], extent[1], extent[2], extent[3]],
                          vmin=0, vmax=vmax, aspect="equal")
    axes[0].set_title(f"Density: {name_a}"); fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(hist_b.T, origin="lower", cmap="Blues",
                          extent=[extent[0], extent[1], extent[2], extent[3]],
                          vmin=0, vmax=vmax, aspect="equal")
    axes[1].set_title(f"Density: {name_b}"); fig.colorbar(im1, ax=axes[1], fraction=0.046)

    rgb = np.zeros((*hist_a.T.shape, 3))
    rgb[:, :, 0] = hist_a.T / max(hist_a.max(), 1)
    rgb[:, :, 2] = hist_b.T / max(hist_b.max(), 1)
    rgb[:, :, 1] = np.minimum(rgb[:, :, 0], rgb[:, :, 2])
    axes[2].imshow(np.clip(rgb, 0, 1), origin="lower",
                   extent=[extent[0], extent[1], extent[2], extent[3]], aspect="equal")
    axes[2].set_title("Merged (white=overlap)")

    axes[3].scatter(ha, hb, s=20, c="grey", alpha=0.5, edgecolors="k", linewidths=0.2)
    axes[3].set_xlabel(f"Counts ({name_a})"); axes[3].set_ylabel(f"Counts ({name_b})")
    axes[3].set_title(f"r={r_pearson:.3f}  ρ={r_spearman:.3f}\nM1={M1:.3f}  M2={M2:.3f}")
    if ha.std() > 0:
        m, b = np.polyfit(ha, hb, 1)
        xr = np.linspace(ha.min(), ha.max(), 100)
        axes[3].plot(xr, m * xr + b, color=COLOR_BOTH, lw=2)

    for ax in axes[:3]:
        add_scale_bar(ax, (extent[1] - extent[0]) * 0.15)

    fig.suptitle("Grid-Based Intensity Correlation", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("03_grid_correlation", run_id),
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Grid correlation analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PROXIMITY NETWORK
# ──────────────────────────────────────────────────────────────────────────────
def proximity_network_analysis(df_a, df_b, name_a, name_b, extent, outdir,
                                run_id, distance_thresholds=None):
    ca, cb = coords(df_a), coords(df_b)
    D = distance_matrix(ca, cb)

    if distance_thresholds is None:
        all_d = D.ravel()
        distance_thresholds = [
            np.percentile(all_d, 5),
            np.percentile(all_d, 10),
            np.percentile(all_d, 25),
        ]

    stats = {}
    fig, axes = plt.subplots(1, len(distance_thresholds),
                              figsize=(8 * len(distance_thresholds), 8), dpi=DPI)
    if len(distance_thresholds) == 1:
        axes = [axes]

    for k, thresh in enumerate(distance_thresholds):
        ax = axes[k]
        G = nx.Graph()
        for i in range(len(ca)):
            G.add_node(f"A_{i}", pos=tuple(ca[i]), group="A")
        for j in range(len(cb)):
            G.add_node(f"B_{j}", pos=tuple(cb[j]), group="B")
        pairs = np.argwhere(D <= thresh)
        for i, j in pairs:
            G.add_edge(f"A_{i}", f"B_{j}", weight=D[i, j])

        n_edges = G.number_of_edges()
        n_comp = nx.number_connected_components(G)
        deg_a = [G.degree(f"A_{i}") for i in range(len(ca))]
        deg_b = [G.degree(f"B_{j}") for j in range(len(cb))]
        frac_a = np.mean(np.array(deg_a) > 0)
        frac_b = np.mean(np.array(deg_b) > 0)

        stats[f"thresh={thresh:.1f}|edges"] = n_edges
        stats[f"thresh={thresh:.1f}|components"] = n_comp
        stats[f"thresh={thresh:.1f}|mean_deg_A"] = np.mean(deg_a)
        stats[f"thresh={thresh:.1f}|mean_deg_B"] = np.mean(deg_b)
        stats[f"thresh={thresh:.1f}|frac_A_conn"] = frac_a
        stats[f"thresh={thresh:.1f}|frac_B_conn"] = frac_b

        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

        edge_seg, edge_w = [], []
        for u, v, d in G.edges(data=True):
            edge_seg.append([G.nodes[u]["pos"], G.nodes[v]["pos"]])
            edge_w.append(d["weight"])
        if edge_seg:
            norm = Normalize(vmin=0, vmax=thresh)
            lc = LineCollection(edge_seg, cmap="YlOrRd_r", norm=norm,
                                linewidths=0.4, alpha=0.35)
            lc.set_array(np.array(edge_w)); ax.add_collection(lc)

        vmax_deg = max(max(deg_a), max(deg_b), 1)
        sc_a = ax.scatter(ca[:, 0], ca[:, 1], c=deg_a, cmap="Reds", s=20,
                          vmin=0, vmax=vmax_deg, edgecolors="k", linewidths=0.3,
                          zorder=4, marker="o")
        ax.scatter(cb[:, 0], cb[:, 1], c=deg_b, cmap="Blues", s=20,
                   vmin=0, vmax=vmax_deg, edgecolors="k", linewidths=0.3,
                   zorder=4, marker="s")
        cbar = fig.colorbar(sc_a, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Degree")
        ax.set_title(f"Thresh={thresh:.1f}\nEdges={n_edges} Comp={n_comp}\n"
                     f"Conn: {name_a}={frac_a:.0%} {name_b}={frac_b:.0%}", fontsize=10)
        legend_el = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_A,
                   markersize=8, label=name_a),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_B,
                   markersize=8, label=name_b),
        ]
        ax.legend(handles=legend_el, loc="upper right", fontsize=8)
        add_scale_bar(ax, (extent[1] - extent[0]) * 0.15)

    fig.suptitle("Proximity Network Analysis (A ↔ B)", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("04_proximity_network", run_id),
                bbox_inches="tight")
    plt.close(fig)

    # degree distribution
    mid_thresh = distance_thresholds[len(distance_thresholds) // 2]
    pairs = np.argwhere(D <= mid_thresh)
    deg_a_arr = np.zeros(len(ca)); deg_b_arr = np.zeros(len(cb))
    for i, j in pairs:
        deg_a_arr[i] += 1; deg_b_arr[j] += 1

    fig2, axes2 = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    max_deg = int(max(deg_a_arr.max(), deg_b_arr.max()))
    bins_deg = np.arange(-0.5, max_deg + 1.5, 1)
    axes2[0].hist(deg_a_arr, bins=bins_deg, color=COLOR_A, alpha=0.7,
                  label=name_a, edgecolor="k", linewidth=0.3)
    axes2[0].hist(deg_b_arr, bins=bins_deg, color=COLOR_B, alpha=0.7,
                  label=name_b, edgecolor="k", linewidth=0.3)
    axes2[0].set_xlabel("Degree"); axes2[0].set_ylabel("Count")
    axes2[0].set_title(f"Degree Distribution (thresh={mid_thresh:.1f})")
    axes2[0].legend()

    axes2[1].scatter(df_a["area"], deg_a_arr, c=COLOR_A, alpha=0.5, s=15, label=name_a)
    axes2[1].scatter(df_b["area"], deg_b_arr, c=COLOR_B, alpha=0.5, s=15, label=name_b)
    axes2[1].set_xlabel("Area"); axes2[1].set_ylabel("Degree")
    axes2[1].set_title("Degree vs Area"); axes2[1].legend()

    fig2.suptitle("Network Degree Statistics", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(outdir / stamped_filename("04b_degree_distribution", run_id),
                 bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ Proximity network analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 5.  K-MEANS CO-CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────
def kmeans_analysis(df_a, df_b, name_a, name_b, extent, outdir, run_id,
                    k_range=range(2, 11)):
    ca, cb = coords(df_a), coords(df_b)
    all_pts = np.vstack([ca, cb])
    labels_true = np.array([0] * len(ca) + [1] * len(cb))

    scaler = StandardScaler()
    all_pts_sc = scaler.fit_transform(all_pts)

    inertias, silhouettes, mixing_scores = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        cl = km.fit_predict(all_pts_sc)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(all_pts_sc, cl))
        entropies = []
        for c in range(k):
            mask = cl == c; n_in = mask.sum()
            if n_in == 0: continue
            p_a = (labels_true[mask] == 0).sum() / n_in; p_b = 1 - p_a
            e = -(p_a * np.log2(p_a) + p_b * np.log2(p_b)) if 0 < p_a < 1 else 0
            entropies.append(e)
        mixing_scores.append(np.mean(entropies))

    best_k = list(k_range)[np.argmax(silhouettes)]
    stats = {
        "Best k (silhouette)": best_k,
        "Best silhouette score": max(silhouettes),
        "Mixing entropy at best k": mixing_scores[np.argmax(silhouettes)],
    }

    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    cl_best = km_best.fit_predict(all_pts_sc)
    centers = scaler.inverse_transform(km_best.cluster_centers_)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE, dpi=DPI)
    axes[0, 0].plot(list(k_range), inertias, "o-", color="k")
    axes[0, 0].set_xlabel("k"); axes[0, 0].set_ylabel("Inertia")
    axes[0, 0].set_title("Elbow Plot")

    axes[0, 1].plot(list(k_range), silhouettes, "s-", color=COLOR_BOTH)
    axes[0, 1].axvline(best_k, ls="--", color="grey")
    axes[0, 1].set_xlabel("k"); axes[0, 1].set_ylabel("Silhouette")
    axes[0, 1].set_title("Silhouette Score")

    axes[1, 0].bar(list(k_range), mixing_scores, color=COLOR_BOTH, alpha=0.7, edgecolor="k")
    axes[1, 0].set_xlabel("k"); axes[1, 0].set_ylabel("Mean Mixing Entropy")
    axes[1, 0].set_title("A/B Mixing (1=perfect)"); axes[1, 0].set_ylim(0, 1.05)

    ax = axes[1, 1]
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
    cmap_k = cm.get_cmap("tab10", best_k)
    for c in range(best_k):
        mask = cl_best == c
        mask_a = mask & (labels_true == 0); mask_b = mask & (labels_true == 1)
        col = cmap_k(c)
        ax.scatter(all_pts[mask_a, 0], all_pts[mask_a, 1], c=[col], marker="o", s=18, alpha=0.6,
                   edgecolors="k", linewidths=0.2)
        ax.scatter(all_pts[mask_b, 0], all_pts[mask_b, 1], c=[col], marker="s", s=18, alpha=0.6,
                   edgecolors="k", linewidths=0.2)
    ax.scatter(centers[:, 0], centers[:, 1], c="k", marker="X", s=150,
               edgecolors="white", linewidths=1.5, zorder=10)
    legend_el = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label=name_a),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', markersize=8, label=name_b),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='k', markersize=10, label="Centres"),
    ]
    ax.legend(handles=legend_el, fontsize=9, loc="upper right")
    ax.set_title(f"K-Means (k={best_k})")
    add_scale_bar(ax, (extent[1] - extent[0]) * 0.15)

    fig.suptitle("K-Means Co-Clustering Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("05_kmeans", run_id), bbox_inches="tight")
    plt.close(fig)

    # composition bar
    fig3, ax3 = plt.subplots(figsize=(max(6, best_k * 1.2), 5), dpi=DPI)
    cc = np.zeros((best_k, 2))
    for c in range(best_k):
        mask = cl_best == c
        cc[c, 0] = (labels_true[mask] == 0).sum()
        cc[c, 1] = (labels_true[mask] == 1).sum()
    fa = cc[:, 0] / cc.sum(axis=1); fb = cc[:, 1] / cc.sum(axis=1)
    x_pos = np.arange(best_k)
    ax3.bar(x_pos, fa, color=COLOR_A, label=name_a, edgecolor="k", lw=0.3)
    ax3.bar(x_pos, fb, bottom=fa, color=COLOR_B, label=name_b, edgecolor="k", lw=0.3)
    ax3.set_xticks(x_pos); ax3.set_xticklabels([f"C{c}" for c in range(best_k)])
    ax3.set_ylabel("Fraction"); ax3.set_title(f"Cluster Composition (k={best_k})")
    ax3.legend(); fig3.tight_layout()
    fig3.savefig(outdir / stamped_filename("05b_kmeans_composition", run_id),
                 bbox_inches="tight")
    plt.close(fig3)
    print("  ✓ K-Means co-clustering saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 6.  DBSCAN
# ──────────────────────────────────────────────────────────────────────────────
def dbscan_analysis(df_a, df_b, name_a, name_b, extent, outdir, run_id,
                    eps=None, min_samples=5):
    ca, cb = coords(df_a), coords(df_b)
    all_pts = np.vstack([ca, cb])
    labels_true = np.array([0] * len(ca) + [1] * len(cb))

    if eps is None:
        tree = KDTree(all_pts)
        d5, _ = tree.query(all_pts, k=min_samples + 1)
        eps = np.median(d5[:, -1])

    db = DBSCAN(eps=eps, min_samples=min_samples)
    cl = db.fit_predict(all_pts)
    n_clusters = len(set(cl) - {-1}); n_noise = (cl == -1).sum()

    mixed = 0
    for c in set(cl) - {-1}:
        mask = cl == c
        if (labels_true[mask] == 0).any() and (labels_true[mask] == 1).any():
            mixed += 1

    stats = {
        "DBSCAN eps": eps, "DBSCAN min_samples": min_samples,
        "DBSCAN n_clusters": n_clusters, "DBSCAN n_noise": n_noise,
        "DBSCAN mixed_clusters": mixed,
        "DBSCAN fraction_mixed": mixed / max(n_clusters, 1),
    }

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    for ax in axes:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

    unique_cl = sorted(set(cl) - {-1})
    cmap_db = cm.get_cmap("tab20", max(len(unique_cl), 1))
    for idx, c in enumerate(unique_cl):
        mask = cl == c
        axes[0].scatter(all_pts[mask, 0], all_pts[mask, 1], c=[cmap_db(idx)], s=18,
                        alpha=0.7, edgecolors="k", linewidths=0.2,
                        label=f"C{c}" if idx < 10 else None)
    noise_m = cl == -1
    axes[0].scatter(all_pts[noise_m, 0], all_pts[noise_m, 1], c="lightgrey",
                    s=8, alpha=0.4, marker="x", label="Noise")
    axes[0].set_title(f"DBSCAN (eps={eps:.1f})\n{n_clusters} clusters, {n_noise} noise")
    if n_clusters <= 10: axes[0].legend(fontsize=7, loc="upper right", ncol=2)

    for idx, c in enumerate(unique_cl):
        mask = cl == c
        is_mixed = (labels_true[mask] == 0).any() and (labels_true[mask] == 1).any()
        col = COLOR_BOTH if is_mixed else "lightgrey"
        alpha = 0.8 if is_mixed else 0.3
        pts_c = all_pts[mask]
        axes[1].scatter(pts_c[:, 0], pts_c[:, 1], c=col, s=18, alpha=alpha,
                        edgecolors="k", linewidths=0.2)
        if is_mixed and len(pts_c) >= 3:
            hull = ConvexHull(pts_c)
            for simplex in hull.simplices:
                axes[1].plot(pts_c[simplex, 0], pts_c[simplex, 1],
                             color=COLOR_BOTH, lw=1.5, alpha=0.7)
    axes[1].scatter([], [], c=COLOR_BOTH, s=40, label="Mixed")
    axes[1].scatter([], [], c="lightgrey", s=40, label="Single/Noise")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_title(f"Mixed Clusters: {mixed}/{n_clusters}")

    for ax in axes:
        add_scale_bar(ax, (extent[1] - extent[0]) * 0.15)

    fig.suptitle("DBSCAN Density-Based Colocalization", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("06_dbscan", run_id), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ DBSCAN analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 7.  RIPLEY'S CROSS-K / CROSS-L
# ──────────────────────────────────────────────────────────────────────────────
def ripleys_cross_k(df_a, df_b, name_a, name_b, extent, outdir, run_id,
                    n_r=50, n_sim=99):
    ca, cb = coords(df_a), coords(df_b)
    area_roi = (extent[1] - extent[0]) * (extent[3] - extent[2])
    n_a, n_b = len(ca), len(cb)
    lambda_b = n_b / area_roi
    r_max = min(extent[1] - extent[0], extent[3] - extent[2]) * 0.25
    r_vals = np.linspace(0, r_max, n_r)

    tree_b = KDTree(cb)
    K_obs = np.zeros(n_r)
    for i, r in enumerate(r_vals):
        counts = tree_b.query_ball_point(ca, r)
        K_obs[i] = sum(len(c) for c in counts) / (n_a * lambda_b)
    L_obs = np.sqrt(K_obs / np.pi) - r_vals

    K_sims = np.zeros((n_sim, n_r))
    for s in range(n_sim):
        sim = np.column_stack([
            np.random.uniform(extent[0], extent[1], n_b),
            np.random.uniform(extent[2], extent[3], n_b)])
        tree_sim = KDTree(sim)
        for i, r in enumerate(r_vals):
            counts = tree_sim.query_ball_point(ca, r)
            K_sims[s, i] = sum(len(c) for c in counts) / (n_a * lambda_b)
    L_sims = np.sqrt(K_sims / np.pi) - r_vals
    env_lo = np.percentile(L_sims, 2.5, axis=0)
    env_hi = np.percentile(L_sims, 97.5, axis=0)
    env_mean = np.mean(L_sims, axis=0)

    stats = {
        "Ripley cross-L max deviation": float(np.max(L_obs - env_mean)),
        "Ripley cross-L mean deviation": float(np.mean(L_obs - env_mean)),
    }

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    axes[0].plot(r_vals, K_obs, color=COLOR_BOTH, lw=2, label="Observed K_ab")
    axes[0].plot(r_vals, np.pi * r_vals**2, "k--", lw=1, label="CSR (πr²)")
    axes[0].fill_between(r_vals, np.percentile(K_sims, 2.5, axis=0),
                          np.percentile(K_sims, 97.5, axis=0),
                          color="grey", alpha=0.25, label="95% CSR envelope")
    axes[0].set_xlabel("r"); axes[0].set_ylabel("K_ab(r)")
    axes[0].set_title("Cross-K Function"); axes[0].legend(fontsize=9)

    axes[1].plot(r_vals, L_obs, color=COLOR_BOTH, lw=2, label="Observed L_ab")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].fill_between(r_vals, env_lo, env_hi, color="grey", alpha=0.25,
                          label="95% CSR envelope")
    axes[1].set_xlabel("r"); axes[1].set_ylabel("L_ab(r) − r")
    axes[1].set_title("Cross-L Function (>0 = clustering)"); axes[1].legend(fontsize=9)

    fig.suptitle("Ripley's Cross-K / Cross-L Function", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("07_ripleys_cross_K", run_id),
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Ripley's cross-K/L saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 8.  KDE OVERLAP
# ──────────────────────────────────────────────────────────────────────────────
def kde_overlap_analysis(df_a, df_b, name_a, name_b, extent, outdir, run_id,
                         grid_n=200):
    ca, cb = coords(df_a), coords(df_b)
    xg = np.linspace(extent[0], extent[1], grid_n)
    yg = np.linspace(extent[2], extent[3], grid_n)
    XX, YY = np.meshgrid(xg, yg)
    pos = np.vstack([XX.ravel(), YY.ravel()])

    Za = gaussian_kde(ca.T)(pos).reshape(grid_n, grid_n)
    Zb = gaussian_kde(cb.T)(pos).reshape(grid_n, grid_n)
    cell_area = (xg[1] - xg[0]) * (yg[1] - yg[0])
    Za /= Za.sum() * cell_area; Zb /= Zb.sum() * cell_area

    overlap = float(np.sum(np.sqrt(Za * Zb)) * cell_area)
    min_ov = float(np.sum(np.minimum(Za, Zb)) * cell_area)

    stats = {"KDE Bhattacharyya overlap": overlap,
             "KDE min-overlap (Szymkiewicz-Simpson)": min_ov}

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=DPI)
    axes[0].contourf(XX, YY, Za, levels=20, cmap="Reds", alpha=0.7)
    axes[0].contour(XX, YY, Za, levels=5, colors="darkred", linewidths=0.5)
    axes[0].scatter(ca[:, 0], ca[:, 1], s=5, c="k", alpha=0.3)
    axes[0].set_title(f"KDE: {name_a}"); axes[0].set_aspect("equal")

    axes[1].contourf(XX, YY, Zb, levels=20, cmap="Blues", alpha=0.7)
    axes[1].contour(XX, YY, Zb, levels=5, colors="darkblue", linewidths=0.5)
    axes[1].scatter(cb[:, 0], cb[:, 1], s=5, c="k", alpha=0.3)
    axes[1].set_title(f"KDE: {name_b}"); axes[1].set_aspect("equal")

    rgb_ov = np.zeros((grid_n, grid_n, 3))
    za_n = Za / max(Za.max(), 1e-15); zb_n = Zb / max(Zb.max(), 1e-15)
    rgb_ov[:, :, 0] = za_n; rgb_ov[:, :, 2] = zb_n
    rgb_ov[:, :, 1] = np.minimum(za_n, zb_n)
    axes[2].imshow(np.clip(rgb_ov, 0, 1), origin="lower",
                   extent=[extent[0], extent[1], extent[2], extent[3]], aspect="equal")
    axes[2].set_title(f"Overlap\nBhatt={overlap:.4f} MinOv={min_ov:.4f}")

    for ax in axes:
        add_scale_bar(ax, (extent[1] - extent[0]) * 0.15)

    fig.suptitle("KDE Overlap Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("08_kde_overlap", run_id),
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ KDE overlap saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 9.  VORONOI NEIGHBOURHOOD
# ──────────────────────────────────────────────────────────────────────────────
def voronoi_analysis(df_a, df_b, name_a, name_b, extent, outdir, run_id):
    ca, cb = coords(df_a), coords(df_b)
    all_pts = np.vstack([ca, cb])
    labels_true = np.array([0] * len(ca) + [1] * len(cb))
    n = len(all_pts)
    if n < 4:
        print("  ⚠ Too few points for Voronoi; skipping."); return {}

    vor = Voronoi(all_pts)
    neighbours = {i: set() for i in range(n)}
    for (i, j) in vor.ridge_points:
        neighbours[i].add(j); neighbours[j].add(i)

    cross_frac = np.full(n, np.nan)
    for i in range(n):
        if not neighbours[i]: continue
        cross_frac[i] = sum(
            1 for j in neighbours[i] if labels_true[j] != labels_true[i]
        ) / len(neighbours[i])

    mc_a = np.nanmean(cross_frac[:len(ca)])
    mc_b = np.nanmean(cross_frac[len(ca):])
    p_a = len(ca) / n; exp_a = 1 - p_a; exp_b = p_a

    stats = {
        f"Voronoi cross-frac ({name_a})": mc_a,
        f"Voronoi cross-frac ({name_b})": mc_b,
        f"Voronoi expected ({name_a})": exp_a,
        f"Voronoi expected ({name_b})": exp_b,
        f"Voronoi segregation ({name_a})": 1 - mc_a / exp_a if exp_a else np.nan,
        f"Voronoi segregation ({name_b})": 1 - mc_b / exp_b if exp_b else np.nan,
    }

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    ax = axes[0]
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    for idx, (i, j) in enumerate(vor.ridge_points):
        verts = vor.ridge_vertices[idx]
        if -1 in verts: continue
        v0, v1 = vor.vertices[verts[0]], vor.vertices[verts[1]]
        is_cross = labels_true[i] != labels_true[j]
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]],
                color=COLOR_BOTH if is_cross else "lightgrey",
                lw=1.2 if is_cross else 0.4,
                alpha=0.8 if is_cross else 0.3)

    sc = ax.scatter(all_pts[:, 0], all_pts[:, 1], c=cross_frac, cmap="RdYlGn",
                    s=25, edgecolors="k", linewidths=0.3, vmin=0, vmax=1, zorder=5)
    fig.colorbar(sc, ax=ax, fraction=0.046).set_label("Cross-type frac")
    legend_el = [
        Line2D([0], [0], color=COLOR_BOTH, lw=2, label="Cross-type edge"),
        Line2D([0], [0], color="lightgrey", lw=1, label="Same-type edge"),
    ]
    ax.legend(handles=legend_el, fontsize=9, loc="upper right")
    ax.set_title("Voronoi Tessellation")
    add_scale_bar(ax, (extent[1] - extent[0]) * 0.15)

    bins_f = np.linspace(0, 1, 21)
    axes[1].hist(cross_frac[:len(ca)], bins=bins_f, color=COLOR_A, alpha=0.6,
                 label=name_a, edgecolor="k", linewidth=0.3)
    axes[1].hist(cross_frac[len(ca):], bins=bins_f, color=COLOR_B, alpha=0.6,
                 label=name_b, edgecolor="k", linewidth=0.3)
    axes[1].axvline(exp_a, color=COLOR_A, ls="--", lw=1.5,
                    label=f"Random ({name_a})")
    axes[1].axvline(exp_b, color=COLOR_B, ls="--", lw=1.5,
                    label=f"Random ({name_b})")
    axes[1].set_xlabel("Cross-type fraction"); axes[1].set_ylabel("Count")
    axes[1].set_title("Cross-Type Distribution"); axes[1].legend(fontsize=8)

    fig.suptitle("Voronoi Neighbourhood Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / stamped_filename("09_voronoi", run_id),
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Voronoi analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
def save_summary(all_stats: dict, outdir: Path, run_id: str):
    rows = []
    for section, sdict in all_stats.items():
        for metric, value in sdict.items():
            rows.append({"Section": section, "Metric": metric, "Value": value})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / stamped_filename("summary_metrics", run_id, ".csv"),
              index=False)

    print("\n" + "=" * 70)
    print(f"COLOCALIZATION ANALYSIS SUMMARY   (Run: {run_id})")
    print("=" * 70)
    for section, sdict in all_stats.items():
        print(f"\n── {section} ──")
        for metric, value in sdict.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
    print("=" * 70)
    print(f"\nAll outputs saved to:  {outdir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_analysis(csv_a: str, csv_b: str, name_a: str, name_b: str,
                 outdir: str):
    """Run full colocalization pipeline."""
    run_id = generate_run_id()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  Run ID : {run_id}")
    print(f"  CSV A  : {csv_a}")
    print(f"  CSV B  : {csv_b}")
    print(f"  Output : {outdir.resolve()}")
    print(f"{'=' * 70}\n")

    print(f"Loading {csv_a} ...")
    df_a = load_csv(csv_a)
    print(f"  → {len(df_a)} objects")
    print(f"Loading {csv_b} ...")
    df_b = load_csv(csv_b)
    print(f"  → {len(df_b)} objects")

    extent = compute_extent(df_a, df_b)
    print(f"ROI: x=[{extent[0]:.1f}, {extent[1]:.1f}]  "
          f"y=[{extent[2]:.1f}, {extent[3]:.1f}]\n")

    all_stats = {}
    print("Running analyses...")

    plot_spatial_overlay(df_a, df_b, name_a, name_b, extent, outdir, run_id)
    all_stats["Nearest-Neighbour"] = nearest_neighbour_analysis(
        df_a, df_b, name_a, name_b, extent, outdir, run_id)
    all_stats["Grid Correlation"] = grid_correlation_analysis(
        df_a, df_b, name_a, name_b, extent, outdir, run_id)
    all_stats["Proximity Network"] = proximity_network_analysis(
        df_a, df_b, name_a, name_b, extent, outdir, run_id)
    all_stats["K-Means"] = kmeans_analysis(
        df_a, df_b, name_a, name_b, extent, outdir, run_id)
    all_stats["DBSCAN"] = dbscan_analysis(
        df_a, df_b, name_a, name_b, extent, outdir, run_id)
    all_stats["Ripley's Cross-K"] = ripleys_cross_k(
        df_a, df_b, name_a, name_b, extent, outdir, run_id)
    all_stats["KDE Overlap"] = kde_overlap_analysis(
        df_a, df_b, name_a, name_b, extent, outdir, run_id)
    all_stats["Voronoi"] = voronoi_analysis(
        df_a, df_b, name_a, name_b, extent, outdir, run_id)

    save_summary(all_stats, outdir, run_id)
    return run_id


# ──────────────────────────────────────────────────────────────────────────────
# TKINTER GUI
# ──────────────────────────────────────────────────────────────────────────────
def launch_gui():
    """Launch Tkinter GUI for selecting files and parameters."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import threading

    root = tk.Tk()
    root.title("Colocalization Analysis Tool")
    root.resizable(False, False)

    # ── Style ──
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    BG = "#f0f0f0"
    root.configure(bg=BG)

    # ── Variables ──
    var_csv_a = tk.StringVar(value="")
    var_csv_b = tk.StringVar(value="")
    var_name_a = tk.StringVar(value="Population A")
    var_name_b = tk.StringVar(value="Population B")
    var_outdir = tk.StringVar(value="(same as CSV A directory)")
    var_use_csv_dir = tk.BooleanVar(value=True)
    var_status = tk.StringVar(value="Ready — select your files and press Run.")

    # ── Helper: update outdir when CSV A is picked ──
    def update_outdir_from_csv(*_):
        if var_use_csv_dir.get() and var_csv_a.get():
            parent = str(Path(var_csv_a.get()).parent)
            var_outdir.set(parent)

    var_csv_a.trace_add("write", update_outdir_from_csv)
    var_use_csv_dir.trace_add("write", update_outdir_from_csv)

    # ──────────────── FRAMES ────────────────
    main_frame = ttk.Frame(root, padding=20)
    main_frame.grid(row=0, column=0, sticky="nsew")

    # Title
    title_lbl = ttk.Label(main_frame,
                           text="🔬  Colocalization Analysis",
                           font=("Helvetica", 18, "bold"))
    title_lbl.grid(row=0, column=0, columnspan=3, pady=(0, 15))

    subtitle = ttk.Label(main_frame,
                          text="Select two CSV files to analyse spatial colocalization.\n"
                               "Outputs are timestamped — nothing is ever overwritten.",
                          font=("Helvetica", 10))
    subtitle.grid(row=1, column=0, columnspan=3, pady=(0, 15))

    # ── File Selection ──
    file_frame = ttk.LabelFrame(main_frame, text="  Input Files  ", padding=10)
    file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 10))

    def browse_csv(var):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv *.tsv *.txt"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

    # CSV A
    ttk.Label(file_frame, text="CSV File A:").grid(row=0, column=0, sticky="w",
                                                     padx=(0, 5), pady=3)
    entry_a = ttk.Entry(file_frame, textvariable=var_csv_a, width=55)
    entry_a.grid(row=0, column=1, padx=5, pady=3)
    ttk.Button(file_frame, text="Browse…",
               command=lambda: browse_csv(var_csv_a)).grid(
        row=0, column=2, padx=5, pady=3)

    # CSV B
    ttk.Label(file_frame, text="CSV File B:").grid(row=1, column=0, sticky="w",
                                                     padx=(0, 5), pady=3)
    entry_b = ttk.Entry(file_frame, textvariable=var_csv_b, width=55)
    entry_b.grid(row=1, column=1, padx=5, pady=3)
    ttk.Button(file_frame, text="Browse…",
               command=lambda: browse_csv(var_csv_b)).grid(
        row=1, column=2, padx=5, pady=3)

    # ── Labels ──
    label_frame = ttk.LabelFrame(main_frame, text="  Population Labels  ",
                                  padding=10)
    label_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 10))

    ttk.Label(label_frame, text="Name for A:").grid(row=0, column=0, sticky="w",
                                                      padx=(0, 5), pady=3)
    ttk.Entry(label_frame, textvariable=var_name_a, width=30).grid(
        row=0, column=1, padx=5, pady=3, sticky="w")

    ttk.Label(label_frame, text="Name for B:").grid(row=1, column=0, sticky="w",
                                                      padx=(0, 5), pady=3)
    ttk.Entry(label_frame, textvariable=var_name_b, width=30).grid(
        row=1, column=1, padx=5, pady=3, sticky="w")

    # ── Output Directory ──
    out_frame = ttk.LabelFrame(main_frame, text="  Output Directory  ", padding=10)
    out_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 10))

    chk_same = ttk.Checkbutton(out_frame,
                                text="Same directory as CSV A",
                                variable=var_use_csv_dir)
    chk_same.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))

    ttk.Label(out_frame, text="Directory:").grid(row=1, column=0, sticky="w",
                                                   padx=(0, 5), pady=3)
    entry_out = ttk.Entry(out_frame, textvariable=var_outdir, width=55)
    entry_out.grid(row=1, column=1, padx=5, pady=3)

    def browse_outdir():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            var_use_csv_dir.set(False)
            var_outdir.set(d)

    ttk.Button(out_frame, text="Browse…",
               command=browse_outdir).grid(row=1, column=2, padx=5, pady=3)

    # ── Progress / Status ──
    status_frame = ttk.Frame(main_frame)
    status_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(5, 5))

    progress = ttk.Progressbar(status_frame, mode="indeterminate", length=500)
    progress.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))

    status_lbl = ttk.Label(status_frame, textvariable=var_status,
                            font=("Helvetica", 10, "italic"),
                            foreground="#555555")
    status_lbl.grid(row=1, column=0, columnspan=3, sticky="w")

    # ── Buttons ──
    btn_frame = ttk.Frame(main_frame)
    btn_frame.grid(row=6, column=0, columnspan=3, pady=(10, 0))

    def validate_inputs():
        if not var_csv_a.get().strip():
            messagebox.showerror("Missing Input", "Please select CSV File A.")
            return False
        if not var_csv_b.get().strip():
            messagebox.showerror("Missing Input", "Please select CSV File B.")
            return False
        if not Path(var_csv_a.get()).is_file():
            messagebox.showerror("File Not Found",
                                 f"Cannot find:\n{var_csv_a.get()}")
            return False
        if not Path(var_csv_b.get()).is_file():
            messagebox.showerror("File Not Found",
                                 f"Cannot find:\n{var_csv_b.get()}")
            return False
        if var_use_csv_dir.get():
            var_outdir.set(str(Path(var_csv_a.get()).parent))
        if not var_outdir.get().strip():
            messagebox.showerror("Missing Output",
                                 "Please select an output directory.")
            return False
        return True

    def run_in_thread():
        """Run analysis in background thread to keep GUI responsive."""
        if not validate_inputs():
            return

        csv_a = var_csv_a.get().strip()
        csv_b = var_csv_b.get().strip()
        name_a = var_name_a.get().strip() or "Population A"
        name_b = var_name_b.get().strip() or "Population B"
        outdir = var_outdir.get().strip()

        # disable run button
        btn_run.configure(state="disabled")
        progress.start(15)
        var_status.set("⏳ Analysis running — please wait...")

        def worker():
            try:
                rid = run_analysis(csv_a, csv_b, name_a, name_b, outdir)
                root.after(0, lambda: on_success(rid, outdir))
            except Exception as e:
                root.after(0, lambda: on_error(str(e)))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def on_success(rid, outdir):
        progress.stop()
        btn_run.configure(state="normal")
        var_status.set(f"✅ Done!  Run ID: {rid}")
        messagebox.showinfo(
            "Analysis Complete",
            f"All outputs saved to:\n{Path(outdir).resolve()}\n\n"
            f"Run ID: {rid}\n"
            f"Files are timestamped and will not be overwritten."
        )

    def on_error(err_msg):
        progress.stop()
        btn_run.configure(state="normal")
        var_status.set(f"❌ Error: {err_msg}")
        messagebox.showerror("Analysis Error", f"An error occurred:\n\n{err_msg}")

    btn_run = ttk.Button(btn_frame, text="▶  Run Analysis",
                          command=run_in_thread)
    btn_run.grid(row=0, column=0, padx=10)

    btn_quit = ttk.Button(btn_frame, text="✕  Quit", command=root.destroy)
    btn_quit.grid(row=0, column=1, padx=10)

    # ── Footer ──
    footer = ttk.Label(main_frame,
                        text="Files: 01–09 figures + summary_metrics.csv  |  "
                             "All timestamped (YYYYMMDD_HHMMSS)",
                        font=("Helvetica", 8), foreground="#999999")
    footer.grid(row=7, column=0, columnspan=3, pady=(15, 0))

    # centre window on screen
    root.update_idletasks()
    w = root.winfo_width(); h = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # If command-line args provided → CLI mode; otherwise → GUI
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--gui"):
        import argparse
        parser = argparse.ArgumentParser(
            description="Colocalization analysis (CLI mode)."
        )
        parser.add_argument("csv_a", help="Path to first CSV")
        parser.add_argument("csv_b", help="Path to second CSV")
        parser.add_argument("--name-a", default="Population A")
        parser.add_argument("--name-b", default="Population B")
        parser.add_argument("-o", "--outdir", default=None,
                            help="Output dir (default: same as csv_a)")
        args = parser.parse_args()
        outdir = args.outdir or str(Path(args.csv_a).parent)
        run_analysis(args.csv_a, args.csv_b, args.name_a, args.name_b, outdir)
    else:
        launch_gui()