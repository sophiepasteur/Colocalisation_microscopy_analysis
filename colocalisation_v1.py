#!/usr/bin/env python3
"""
Comprehensive Colocalization Analysis Tool
===========================================
Analyzes spatial colocalization between two populations from CSV files.

Metrics & Visualizations:
  1. Spatial overlay of both distributions
  2. Nearest-neighbor distance analysis
  3. Grid-based intensity correlation (Pearson, Spearman, Manders)
  4. Proximity network analysis
  5. K-Means co-clustering analysis
  6. DBSCAN density-based colocalization
  7. Ripley's cross-K / cross-L function
  8. Kernel density estimation overlap
  9. Voronoi neighborhood analysis
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
from scipy.stats import pearsonr, spearmanr, gaussian_kde, mannwhitneyu
from scipy.ndimage import gaussian_filter

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import networkx as nx

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
COLOR_A = "#E63946"       # red-ish
COLOR_B = "#457B9D"       # blue-ish
COLOR_BOTH = "#2A9D8F"    # teal (overlap / edge)
FIGSIZE_LARGE = (14, 12)
FIGSIZE_WIDE = (16, 7)
FIGSIZE_SQ = (10, 10)
DPI = 150


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and keep relevant columns."""
    df = pd.read_csv(path, sep=None, engine="python")  # auto-detect delimiter
    needed = {"centroid_x", "centroid_y", "area"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[["centroid_x", "centroid_y", "area"]].copy()


def compute_extent(df_a: pd.DataFrame, df_b: pd.DataFrame, pad_frac=0.05):
    """Compute common bounding box with padding."""
    xs = np.concatenate([df_a["centroid_x"].values, df_b["centroid_x"].values])
    ys = np.concatenate([df_a["centroid_y"].values, df_b["centroid_y"].values])
    pad_x = (xs.max() - xs.min()) * pad_frac
    pad_y = (ys.max() - ys.min()) * pad_frac
    return (
        xs.min() - pad_x, xs.max() + pad_x,
        ys.min() - pad_y, ys.max() + pad_y,
    )


def add_scale_bar(ax, length, label=None, loc="lower right", fontsize=9):
    """Add a scale bar to an axis."""
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
    """Map area values to scatter marker sizes."""
    if areas.max() == areas.min():
        return np.full(len(areas), (min_s + max_s) / 2)
    norm = (areas - areas.min()) / (areas.max() - areas.min())
    return min_s + norm * (max_s - min_s)


def coords(df):
    """Return (N,2) array of x,y."""
    return df[["centroid_x", "centroid_y"]].values


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SPATIAL OVERLAY
# ──────────────────────────────────────────────────────────────────────────────
def plot_spatial_overlay(df_a, df_b, name_a, name_b, extent, outdir):
    """Side-by-side and overlaid scatter of both populations."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=DPI)

    sa = area_to_marker_size(df_a["area"].values)
    sb = area_to_marker_size(df_b["area"].values)

    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        scale_len = (extent[1] - extent[0]) * 0.15
        add_scale_bar(ax, scale_len)

    # individual
    axes[0].scatter(df_a["centroid_x"], df_a["centroid_y"], s=sa,
                    c=COLOR_A, alpha=0.6, edgecolors="k", linewidths=0.3)
    axes[0].set_title(f"{name_a}  (n={len(df_a)})", fontsize=13)

    axes[1].scatter(df_b["centroid_x"], df_b["centroid_y"], s=sb,
                    c=COLOR_B, alpha=0.6, edgecolors="k", linewidths=0.3)
    axes[1].set_title(f"{name_b}  (n={len(df_b)})", fontsize=13)

    # overlay
    axes[2].scatter(df_a["centroid_x"], df_a["centroid_y"], s=sa,
                    c=COLOR_A, alpha=0.5, edgecolors="k", linewidths=0.2,
                    label=name_a)
    axes[2].scatter(df_b["centroid_x"], df_b["centroid_y"], s=sb,
                    c=COLOR_B, alpha=0.5, edgecolors="k", linewidths=0.2,
                    label=name_b)
    axes[2].legend(fontsize=11, loc="upper right")
    axes[2].set_title("Overlay", fontsize=13)

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    fig.suptitle("Spatial Distribution Overview", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "01_spatial_overlay.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Spatial overlay saved.")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  NEAREST-NEIGHBOUR DISTANCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def nearest_neighbour_analysis(df_a, df_b, name_a, name_b, extent, outdir):
    """Cross-NN distances + histogram + spatial map."""
    ca, cb = coords(df_a), coords(df_b)
    tree_a, tree_b = KDTree(ca), KDTree(cb)

    # A→B
    dist_ab, idx_ab = tree_b.query(ca)
    # B→A
    dist_ba, idx_ba = tree_a.query(cb)

    # --- stats ---
    stats = {
        f"{name_a}→{name_b} mean NN dist": np.mean(dist_ab),
        f"{name_a}→{name_b} median NN dist": np.median(dist_ab),
        f"{name_b}→{name_a} mean NN dist": np.mean(dist_ba),
        f"{name_b}→{name_a} median NN dist": np.median(dist_ba),
    }

    # Random expectation for CSR (complete spatial randomness)
    area_roi = (extent[1] - extent[0]) * (extent[3] - extent[2])
    lambda_b = len(df_b) / area_roi
    lambda_a = len(df_a) / area_roi
    expected_ab = 0.5 / np.sqrt(lambda_b) if lambda_b > 0 else np.nan
    expected_ba = 0.5 / np.sqrt(lambda_a) if lambda_a > 0 else np.nan
    stats[f"Expected NN dist (CSR) {name_a}→{name_b}"] = expected_ab
    stats[f"Expected NN dist (CSR) {name_b}→{name_a}"] = expected_ba
    stats[f"Colocalization Index {name_a}→{name_b} (obs/exp)"] = (
        np.mean(dist_ab) / expected_ab if expected_ab else np.nan
    )
    stats[f"Colocalization Index {name_b}→{name_a} (obs/exp)"] = (
        np.mean(dist_ba) / expected_ba if expected_ba else np.nan
    )

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=DPI)

    # histograms
    bins = np.linspace(0, max(dist_ab.max(), dist_ba.max()), 50)
    axes[0].hist(dist_ab, bins=bins, color=COLOR_A, alpha=0.65,
                 label=f"{name_a}→{name_b}", edgecolor="k", linewidth=0.3)
    axes[0].hist(dist_ba, bins=bins, color=COLOR_B, alpha=0.65,
                 label=f"{name_b}→{name_a}", edgecolor="k", linewidth=0.3)
    axes[0].axvline(expected_ab, ls="--", color=COLOR_A, lw=1.5,
                    label=f"CSR expect. {name_a}→{name_b}")
    axes[0].axvline(expected_ba, ls="--", color=COLOR_B, lw=1.5,
                    label=f"CSR expect. {name_b}→{name_a}")
    axes[0].set_xlabel("Nearest-Neighbour Distance")
    axes[0].set_ylabel("Count")
    axes[0].set_title("NN Distance Distributions")
    axes[0].legend(fontsize=8)

    # CDF
    for d, c, lab in [(dist_ab, COLOR_A, f"{name_a}→{name_b}"),
                       (dist_ba, COLOR_B, f"{name_b}→{name_a}")]:
        sorted_d = np.sort(d)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        axes[1].plot(sorted_d, cdf, color=c, lw=2, label=lab)
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Cumulative Fraction")
    axes[1].set_title("Cumulative NN Distance")
    axes[1].legend(fontsize=9)

    # spatial map: lines connecting NN pairs (A→B)
    ax = axes[2]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    # draw connection lines coloured by distance
    norm = Normalize(vmin=0, vmax=np.percentile(dist_ab, 95))
    segments = []
    colors_seg = []
    for i, j in enumerate(idx_ab):
        segments.append([ca[i], cb[j]])
        colors_seg.append(dist_ab[i])
    lc = LineCollection(segments, cmap="viridis_r", norm=norm,
                        linewidths=0.6, alpha=0.5)
    lc.set_array(np.array(colors_seg))
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NN Distance")

    ax.scatter(ca[:, 0], ca[:, 1], s=12, c=COLOR_A, zorder=3, label=name_a)
    ax.scatter(cb[:, 0], cb[:, 1], s=12, c=COLOR_B, zorder=3, label=name_b)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"NN Links  {name_a}→{name_b}")
    scale_len = (extent[1] - extent[0]) * 0.15
    add_scale_bar(ax, scale_len)

    fig.suptitle("Nearest-Neighbour Distance Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "02_nearest_neighbour.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Nearest-neighbour analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 3.  GRID-BASED INTENSITY CORRELATION  (Pearson, Spearman, Manders)
# ──────────────────────────────────────────────────────────────────────────────
def grid_correlation_analysis(df_a, df_b, name_a, name_b, extent, outdir,
                              n_bins=30):
    """Bin space into a grid; compute density per bin; correlate."""
    x_edges = np.linspace(extent[0], extent[1], n_bins + 1)
    y_edges = np.linspace(extent[2], extent[3], n_bins + 1)

    hist_a, _, _ = np.histogram2d(
        df_a["centroid_x"], df_a["centroid_y"], bins=[x_edges, y_edges]
    )
    hist_b, _, _ = np.histogram2d(
        df_b["centroid_x"], df_b["centroid_y"], bins=[x_edges, y_edges]
    )

    ha = hist_a.ravel().astype(float)
    hb = hist_b.ravel().astype(float)

    # correlations
    r_pearson, p_pearson = pearsonr(ha, hb)
    r_spearman, p_spearman = spearmanr(ha, hb)

    # Manders-like coefficients (fraction of A that overlaps B and vice versa)
    mask_b = hb > 0
    mask_a = ha > 0
    M1 = ha[mask_b].sum() / ha.sum() if ha.sum() > 0 else 0  # fraction of A in B-positive bins
    M2 = hb[mask_a].sum() / hb.sum() if hb.sum() > 0 else 0

    # Costes randomisation p-value (simplified: 200 shuffles)
    n_rand = 200
    rand_r = np.zeros(n_rand)
    for i in range(n_rand):
        np.random.shuffle(hb_shuffled := hb.copy())
        rand_r[i], _ = pearsonr(ha, hb_shuffled)
    costes_p = np.mean(rand_r >= r_pearson)

    stats = {
        "Pearson r": r_pearson,
        "Pearson p-value": p_pearson,
        "Spearman ρ": r_spearman,
        "Spearman p-value": p_spearman,
        f"Manders M1 (frac {name_a} in {name_b} regions)": M1,
        f"Manders M2 (frac {name_b} in {name_a} regions)": M2,
        "Costes randomisation p (Pearson)": costes_p,
    }

    # --- figure ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=DPI)

    # density maps
    vmax = max(hist_a.max(), hist_b.max())
    im0 = axes[0].imshow(hist_a.T, origin="lower", cmap="Reds",
                          extent=[extent[0], extent[1], extent[2], extent[3]],
                          vmin=0, vmax=vmax, aspect="equal")
    axes[0].set_title(f"Density: {name_a}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(hist_b.T, origin="lower", cmap="Blues",
                          extent=[extent[0], extent[1], extent[2], extent[3]],
                          vmin=0, vmax=vmax, aspect="equal")
    axes[1].set_title(f"Density: {name_b}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # merged (green = overlap)
    rgb = np.zeros((*hist_a.T.shape, 3))
    rgb[:, :, 0] = hist_a.T / max(hist_a.max(), 1)  # red channel
    rgb[:, :, 2] = hist_b.T / max(hist_b.max(), 1)  # blue channel
    rgb[:, :, 1] = np.minimum(rgb[:, :, 0], rgb[:, :, 2])  # green = overlap
    axes[2].imshow(np.clip(rgb, 0, 1), origin="lower",
                   extent=[extent[0], extent[1], extent[2], extent[3]],
                   aspect="equal")
    axes[2].set_title("Merged (magenta/cyan → white=overlap)")

    # scatter-correlation
    axes[3].scatter(ha, hb, s=20, c="grey", alpha=0.5, edgecolors="k",
                    linewidths=0.2)
    axes[3].set_xlabel(f"Counts per bin ({name_a})")
    axes[3].set_ylabel(f"Counts per bin ({name_b})")
    axes[3].set_title(
        f"r={r_pearson:.3f}  ρ={r_spearman:.3f}\n"
        f"M1={M1:.3f}  M2={M2:.3f}"
    )
    # regression line
    if ha.std() > 0:
        m, b = np.polyfit(ha, hb, 1)
        xr = np.linspace(ha.min(), ha.max(), 100)
        axes[3].plot(xr, m * xr + b, color=COLOR_BOTH, lw=2)

    for ax in axes[:3]:
        scale_len = (extent[1] - extent[0]) * 0.15
        add_scale_bar(ax, scale_len)

    fig.suptitle("Grid-Based Intensity Correlation", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "03_grid_correlation.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Grid correlation analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PROXIMITY NETWORK
# ──────────────────────────────────────────────────────────────────────────────
def proximity_network_analysis(df_a, df_b, name_a, name_b, extent, outdir,
                                distance_thresholds=None):
    """
    Build a bipartite proximity network: edges connect A↔B nodes
    within a threshold distance.  Analyse degree distribution and
    connected components.
    """
    ca, cb = coords(df_a), coords(df_b)
    D = distance_matrix(ca, cb)

    if distance_thresholds is None:
        # auto: 10th, 25th, 50th percentile of all pairwise distances
        all_d = D.ravel()
        distance_thresholds = [
            np.percentile(all_d, 5),
            np.percentile(all_d, 10),
            np.percentile(all_d, 25),
        ]

    stats = {}

    fig, axes = plt.subplots(1, len(distance_thresholds),
                              figsize=(8 * len(distance_thresholds), 8),
                              dpi=DPI)
    if len(distance_thresholds) == 1:
        axes = [axes]

    for k, thresh in enumerate(distance_thresholds):
        ax = axes[k]
        G = nx.Graph()
        # nodes
        for i in range(len(ca)):
            G.add_node(f"A_{i}", pos=tuple(ca[i]), group="A")
        for j in range(len(cb)):
            G.add_node(f"B_{j}", pos=tuple(cb[j]), group="B")
        # edges
        pairs = np.argwhere(D <= thresh)
        for i, j in pairs:
            G.add_edge(f"A_{i}", f"B_{j}", weight=D[i, j])

        n_edges = G.number_of_edges()
        n_components = nx.number_connected_components(G)
        degrees_a = [G.degree(f"A_{i}") for i in range(len(ca))]
        degrees_b = [G.degree(f"B_{j}") for j in range(len(cb))]
        frac_a_connected = np.mean(np.array(degrees_a) > 0)
        frac_b_connected = np.mean(np.array(degrees_b) > 0)

        stats[f"thresh={thresh:.1f} | edges"] = n_edges
        stats[f"thresh={thresh:.1f} | components"] = n_components
        stats[f"thresh={thresh:.1f} | mean_deg_A"] = np.mean(degrees_a)
        stats[f"thresh={thresh:.1f} | mean_deg_B"] = np.mean(degrees_b)
        stats[f"thresh={thresh:.1f} | frac_A_connected"] = frac_a_connected
        stats[f"thresh={thresh:.1f} | frac_B_connected"] = frac_b_connected

        # plot
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

        # draw edges
        edge_segments = []
        edge_weights = []
        for u, v, d in G.edges(data=True):
            p1 = G.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            edge_segments.append([p1, p2])
            edge_weights.append(d["weight"])
        if edge_segments:
            norm = Normalize(vmin=0, vmax=thresh)
            lc = LineCollection(edge_segments, cmap="YlOrRd_r", norm=norm,
                                linewidths=0.4, alpha=0.35)
            lc.set_array(np.array(edge_weights))
            ax.add_collection(lc)

        # draw nodes  — colour by degree
        deg_a = np.array(degrees_a, dtype=float)
        deg_b = np.array(degrees_b, dtype=float)
        vmax_deg = max(deg_a.max(), deg_b.max(), 1)

        sc_a = ax.scatter(ca[:, 0], ca[:, 1], c=deg_a, cmap="Reds",
                          s=20, vmin=0, vmax=vmax_deg, edgecolors="k",
                          linewidths=0.3, zorder=4, marker="o")
        sc_b = ax.scatter(cb[:, 0], cb[:, 1], c=deg_b, cmap="Blues",
                          s=20, vmin=0, vmax=vmax_deg, edgecolors="k",
                          linewidths=0.3, zorder=4, marker="s")
        cbar = fig.colorbar(sc_a, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Degree (cross-type)")

        ax.set_title(
            f"Threshold = {thresh:.1f}\n"
            f"Edges={n_edges}  Components={n_components}\n"
            f"Connected: {name_a}={frac_a_connected:.0%}  "
            f"{name_b}={frac_b_connected:.0%}",
            fontsize=10,
        )
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_A,
                   markersize=8, label=name_a),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_B,
                   markersize=8, label=name_b),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        scale_len = (extent[1] - extent[0]) * 0.15
        add_scale_bar(ax, scale_len)

    fig.suptitle("Proximity Network Analysis (A ↔ B)", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "04_proximity_network.png", bbox_inches="tight")
    plt.close(fig)

    # ---- degree distribution figure ----
    fig2, axes2 = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    # Use the middle threshold for the degree histogram
    mid_thresh = distance_thresholds[len(distance_thresholds) // 2]
    pairs = np.argwhere(D <= mid_thresh)
    deg_a_arr = np.zeros(len(ca))
    deg_b_arr = np.zeros(len(cb))
    for i, j in pairs:
        deg_a_arr[i] += 1
        deg_b_arr[j] += 1

    max_deg = int(max(deg_a_arr.max(), deg_b_arr.max()))
    bins_deg = np.arange(-0.5, max_deg + 1.5, 1)
    axes2[0].hist(deg_a_arr, bins=bins_deg, color=COLOR_A, alpha=0.7,
                  label=name_a, edgecolor="k", linewidth=0.3)
    axes2[0].hist(deg_b_arr, bins=bins_deg, color=COLOR_B, alpha=0.7,
                  label=name_b, edgecolor="k", linewidth=0.3)
    axes2[0].set_xlabel("Degree (cross-type neighbours)")
    axes2[0].set_ylabel("Count")
    axes2[0].set_title(f"Degree Distribution (threshold={mid_thresh:.1f})")
    axes2[0].legend()

    # scatter: degree A vs area A
    axes2[1].scatter(df_a["area"], deg_a_arr, c=COLOR_A, alpha=0.5, s=15,
                     label=name_a)
    axes2[1].scatter(df_b["area"], deg_b_arr, c=COLOR_B, alpha=0.5, s=15,
                     label=name_b)
    axes2[1].set_xlabel("Area")
    axes2[1].set_ylabel("Degree")
    axes2[1].set_title("Degree vs Area")
    axes2[1].legend()

    fig2.suptitle("Network Degree Statistics", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(outdir / "04b_degree_distribution.png", bbox_inches="tight")
    plt.close(fig2)

    print("  ✓ Proximity network analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 5.  K-MEANS CO-CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────
def kmeans_analysis(df_a, df_b, name_a, name_b, extent, outdir,
                    k_range=range(2, 11)):
    """
    Pool both populations; run K-means for several k; measure mixing
    of A and B within each cluster.
    """
    ca, cb = coords(df_a), coords(df_b)
    all_pts = np.vstack([ca, cb])
    labels_true = np.array([0] * len(ca) + [1] * len(cb))

    scaler = StandardScaler()
    all_pts_sc = scaler.fit_transform(all_pts)

    inertias = []
    silhouettes = []
    mixing_scores = []  # entropy-based mixing per k

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        cl = km.fit_predict(all_pts_sc)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(all_pts_sc, cl))

        # mixing: for each cluster compute proportion of A and B
        # then Shannon entropy normalised by log(2)
        entropies = []
        for c in range(k):
            mask = cl == c
            n_in = mask.sum()
            if n_in == 0:
                continue
            p_a = (labels_true[mask] == 0).sum() / n_in
            p_b = 1 - p_a
            # entropy
            e = 0
            if 0 < p_a < 1:
                e = -(p_a * np.log2(p_a) + p_b * np.log2(p_b))
            entropies.append(e)
        mixing_scores.append(np.mean(entropies))

    best_k_sil = list(k_range)[np.argmax(silhouettes)]
    stats = {
        "Best k (silhouette)": best_k_sil,
        "Best silhouette score": max(silhouettes),
        f"Mixing entropy at best k": mixing_scores[np.argmax(silhouettes)],
    }

    # refit with best k
    km_best = KMeans(n_clusters=best_k_sil, n_init=10, random_state=42)
    cl_best = km_best.fit_predict(all_pts_sc)
    centers = scaler.inverse_transform(km_best.cluster_centers_)

    # --- figures ---
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE, dpi=DPI)

    # elbow
    axes[0, 0].plot(list(k_range), inertias, "o-", color="k")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Inertia")
    axes[0, 0].set_title("Elbow Plot")

    # silhouette
    axes[0, 1].plot(list(k_range), silhouettes, "s-", color=COLOR_BOTH)
    axes[0, 1].axvline(best_k_sil, ls="--", color="grey")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].set_title("Silhouette Score")

    # mixing
    axes[1, 0].bar(list(k_range), mixing_scores, color=COLOR_BOTH, alpha=0.7,
                   edgecolor="k")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Mean Mixing Entropy")
    axes[1, 0].set_title("A/B Mixing per k  (1=perfectly mixed)")
    axes[1, 0].set_ylim(0, 1.05)

    # spatial clusters at best k
    ax = axes[1, 1]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    cmap_k = cm.get_cmap("tab10", best_k_sil)

    for c in range(best_k_sil):
        mask = cl_best == c
        mask_a = mask & (labels_true == 0)
        mask_b = mask & (labels_true == 1)
        col = cmap_k(c)
        ax.scatter(all_pts[mask_a, 0], all_pts[mask_a, 1], c=[col],
                   marker="o", s=18, alpha=0.6, edgecolors="k", linewidths=0.2)
        ax.scatter(all_pts[mask_b, 0], all_pts[mask_b, 1], c=[col],
                   marker="s", s=18, alpha=0.6, edgecolors="k", linewidths=0.2)

    # cluster centres
    ax.scatter(centers[:, 0], centers[:, 1], c="k", marker="X", s=150,
               edgecolors="white", linewidths=1.5, zorder=10, label="Centres")
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markersize=8, label=name_a),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
               markersize=8, label=name_b),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='k',
               markersize=10, label="Centres"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax.set_title(f"K-Means Clusters  (k={best_k_sil})")
    scale_len = (extent[1] - extent[0]) * 0.15
    add_scale_bar(ax, scale_len)

    fig.suptitle("K-Means Co-Clustering Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "05_kmeans.png", bbox_inches="tight")
    plt.close(fig)

    # --- per-cluster composition bar chart ---
    fig3, ax3 = plt.subplots(figsize=(max(6, best_k_sil * 1.2), 5), dpi=DPI)
    cluster_counts = np.zeros((best_k_sil, 2))
    for c in range(best_k_sil):
        mask = cl_best == c
        cluster_counts[c, 0] = (labels_true[mask] == 0).sum()
        cluster_counts[c, 1] = (labels_true[mask] == 1).sum()
    frac_a = cluster_counts[:, 0] / cluster_counts.sum(axis=1)
    frac_b = cluster_counts[:, 1] / cluster_counts.sum(axis=1)
    x_pos = np.arange(best_k_sil)
    ax3.bar(x_pos, frac_a, color=COLOR_A, label=name_a, edgecolor="k", lw=0.3)
    ax3.bar(x_pos, frac_b, bottom=frac_a, color=COLOR_B, label=name_b,
            edgecolor="k", lw=0.3)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"C{c}" for c in range(best_k_sil)])
    ax3.set_ylabel("Fraction")
    ax3.set_title(f"Cluster Composition  (k={best_k_sil})")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(outdir / "05b_kmeans_composition.png", bbox_inches="tight")
    plt.close(fig3)

    print("  ✓ K-Means co-clustering analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 6.  DBSCAN DENSITY-BASED COLOCALIZATION
# ──────────────────────────────────────────────────────────────────────────────
def dbscan_analysis(df_a, df_b, name_a, name_b, extent, outdir, eps=None,
                    min_samples=5):
    """DBSCAN on pooled data; measure mixed clusters."""
    ca, cb = coords(df_a), coords(df_b)
    all_pts = np.vstack([ca, cb])
    labels_true = np.array([0] * len(ca) + [1] * len(cb))

    if eps is None:
        # heuristic: median of 5th-NN distance
        tree = KDTree(all_pts)
        d5, _ = tree.query(all_pts, k=min_samples + 1)
        eps = np.median(d5[:, -1])

    db = DBSCAN(eps=eps, min_samples=min_samples)
    cl = db.fit_predict(all_pts)
    n_clusters = len(set(cl) - {-1})
    n_noise = (cl == -1).sum()

    # mixed cluster stats
    mixed = 0
    for c in set(cl) - {-1}:
        mask = cl == c
        has_a = (labels_true[mask] == 0).any()
        has_b = (labels_true[mask] == 1).any()
        if has_a and has_b:
            mixed += 1

    stats = {
        "DBSCAN eps": eps,
        "DBSCAN min_samples": min_samples,
        "DBSCAN n_clusters": n_clusters,
        "DBSCAN n_noise": n_noise,
        "DBSCAN mixed_clusters": mixed,
        "DBSCAN fraction_mixed": mixed / max(n_clusters, 1),
    }

    # figure
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

    # left: coloured by cluster
    unique_cl = sorted(set(cl) - {-1})
    cmap_db = cm.get_cmap("tab20", max(len(unique_cl), 1))
    for idx, c in enumerate(unique_cl):
        mask = cl == c
        ax = axes[0]
        ax.scatter(all_pts[mask, 0], all_pts[mask, 1],
                   c=[cmap_db(idx)], s=18, alpha=0.7, edgecolors="k",
                   linewidths=0.2, label=f"C{c}" if idx < 10 else None)
    noise_mask = cl == -1
    axes[0].scatter(all_pts[noise_mask, 0], all_pts[noise_mask, 1],
                    c="lightgrey", s=8, alpha=0.4, marker="x", label="Noise")
    axes[0].set_title(
        f"DBSCAN Clusters  (eps={eps:.1f}, k={min_samples})\n"
        f"{n_clusters} clusters, {n_noise} noise pts"
    )
    if n_clusters <= 10:
        axes[0].legend(fontsize=7, loc="upper right", ncol=2)

    # right: highlight mixed clusters
    for idx, c in enumerate(unique_cl):
        mask = cl == c
        has_a = (labels_true[mask] == 0).any()
        has_b = (labels_true[mask] == 1).any()
        is_mixed = has_a and has_b
        col = COLOR_BOTH if is_mixed else "lightgrey"
        alpha = 0.8 if is_mixed else 0.3
        # convex hull for mixed
        pts_c = all_pts[mask]
        axes[1].scatter(pts_c[:, 0], pts_c[:, 1], c=col, s=18,
                        alpha=alpha, edgecolors="k", linewidths=0.2)
        if is_mixed and len(pts_c) >= 3:
            hull = ConvexHull(pts_c)
            for simplex in hull.simplices:
                axes[1].plot(pts_c[simplex, 0], pts_c[simplex, 1],
                             color=COLOR_BOTH, lw=1.5, alpha=0.7)

    axes[1].scatter([], [], c=COLOR_BOTH, s=40, label="Mixed cluster")
    axes[1].scatter([], [], c="lightgrey", s=40, label="Single-type / Noise")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_title(f"Mixed Clusters: {mixed}/{n_clusters}")

    for ax in axes:
        scale_len = (extent[1] - extent[0]) * 0.15
        add_scale_bar(ax, scale_len)

    fig.suptitle("DBSCAN Density-Based Colocalization", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "06_dbscan.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ DBSCAN analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 7.  RIPLEY'S CROSS-K  /  CROSS-L FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def ripleys_cross_k(df_a, df_b, name_a, name_b, extent, outdir, n_r=50,
                    n_sim=99):
    """
    Estimate cross-K (and cross-L) function and compare with CSR envelope.
    """
    ca, cb = coords(df_a), coords(df_b)
    area_roi = (extent[1] - extent[0]) * (extent[3] - extent[2])
    n_a, n_b = len(ca), len(cb)
    lambda_b = n_b / area_roi

    r_max = min(extent[1] - extent[0], extent[3] - extent[2]) * 0.25
    r_vals = np.linspace(0, r_max, n_r)

    # observed K_ab
    tree_b = KDTree(cb)
    K_obs = np.zeros(n_r)
    for i, r in enumerate(r_vals):
        counts = tree_b.query_ball_point(ca, r)
        K_obs[i] = sum(len(c) for c in counts) / (n_a * lambda_b)

    L_obs = np.sqrt(K_obs / np.pi) - r_vals

    # CSR simulations
    K_sims = np.zeros((n_sim, n_r))
    for s in range(n_sim):
        sim_x = np.random.uniform(extent[0], extent[1], n_b)
        sim_y = np.random.uniform(extent[2], extent[3], n_b)
        sim_pts = np.column_stack([sim_x, sim_y])
        tree_sim = KDTree(sim_pts)
        for i, r in enumerate(r_vals):
            counts = tree_sim.query_ball_point(ca, r)
            K_sims[s, i] = sum(len(c) for c in counts) / (n_a * lambda_b)
    L_sims = np.sqrt(K_sims / np.pi) - r_vals

    env_lo = np.percentile(L_sims, 2.5, axis=0)
    env_hi = np.percentile(L_sims, 97.5, axis=0)
    env_mean = np.mean(L_sims, axis=0)

    stats = {
        "Ripley cross-L max deviation": np.max(L_obs - env_mean),
        "Ripley cross-L mean deviation": np.mean(L_obs - env_mean),
    }

    # figure
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)

    axes[0].plot(r_vals, K_obs, color=COLOR_BOTH, lw=2, label="Observed K_ab")
    axes[0].plot(r_vals, np.pi * r_vals ** 2, "k--", lw=1, label="CSR (πr²)")
    axes[0].fill_between(
        r_vals,
        np.percentile(K_sims, 2.5, axis=0),
        np.percentile(K_sims, 97.5, axis=0),
        color="grey", alpha=0.25, label="95% CSR envelope",
    )
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("K_ab(r)")
    axes[0].set_title("Cross-K Function")
    axes[0].legend(fontsize=9)

    axes[1].plot(r_vals, L_obs, color=COLOR_BOTH, lw=2, label="Observed L_ab")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].fill_between(r_vals, env_lo, env_hi, color="grey", alpha=0.25,
                          label="95% CSR envelope")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("L_ab(r) − r")
    axes[1].set_title("Cross-L Function  (>0 = clustering)")
    axes[1].legend(fontsize=9)

    fig.suptitle("Ripley's Cross-K / Cross-L Function", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "07_ripleys_cross_K.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Ripley's cross-K/L analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 8.  KERNEL DENSITY ESTIMATION OVERLAP
# ──────────────────────────────────────────────────────────────────────────────
def kde_overlap_analysis(df_a, df_b, name_a, name_b, extent, outdir,
                         grid_n=200):
    """Gaussian KDE for each population; compute overlap integral."""
    ca, cb = coords(df_a), coords(df_b)

    xgrid = np.linspace(extent[0], extent[1], grid_n)
    ygrid = np.linspace(extent[2], extent[3], grid_n)
    XX, YY = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([XX.ravel(), YY.ravel()])

    kde_a = gaussian_kde(ca.T)
    kde_b = gaussian_kde(cb.T)
    Za = kde_a(positions).reshape(grid_n, grid_n)
    Zb = kde_b(positions).reshape(grid_n, grid_n)

    # normalise to probability densities
    cell_area = (xgrid[1] - xgrid[0]) * (ygrid[1] - ygrid[0])
    Za /= Za.sum() * cell_area
    Zb /= Zb.sum() * cell_area

    # overlap coefficient (Bhattacharyya-like)
    overlap = np.sum(np.sqrt(Za * Zb)) * cell_area
    # min-overlap (Szymkiewicz–Simpson)
    min_overlap = np.sum(np.minimum(Za, Zb)) * cell_area

    stats = {
        "KDE Bhattacharyya overlap": overlap,
        "KDE min-overlap (Szymkiewicz-Simpson)": min_overlap,
    }

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=DPI)
    ext_img = [extent[0], extent[1], extent[2], extent[3]]

    axes[0].contourf(XX, YY, Za, levels=20, cmap="Reds", alpha=0.7)
    axes[0].contour(XX, YY, Za, levels=5, colors="darkred", linewidths=0.5)
    axes[0].scatter(ca[:, 0], ca[:, 1], s=5, c="k", alpha=0.3)
    axes[0].set_title(f"KDE: {name_a}")
    axes[0].set_aspect("equal")

    axes[1].contourf(XX, YY, Zb, levels=20, cmap="Blues", alpha=0.7)
    axes[1].contour(XX, YY, Zb, levels=5, colors="darkblue", linewidths=0.5)
    axes[1].scatter(cb[:, 0], cb[:, 1], s=5, c="k", alpha=0.3)
    axes[1].set_title(f"KDE: {name_b}")
    axes[1].set_aspect("equal")

    # overlap image
    rgb_ov = np.zeros((grid_n, grid_n, 3))
    za_norm = Za / max(Za.max(), 1e-15)
    zb_norm = Zb / max(Zb.max(), 1e-15)
    rgb_ov[:, :, 0] = za_norm
    rgb_ov[:, :, 2] = zb_norm
    rgb_ov[:, :, 1] = np.minimum(za_norm, zb_norm)
    axes[2].imshow(np.clip(rgb_ov, 0, 1), origin="lower", extent=ext_img,
                   aspect="equal")
    axes[2].set_title(
        f"KDE Overlap\nBhattacharyya={overlap:.4f}  "
        f"Min-overlap={min_overlap:.4f}"
    )

    for ax in axes:
        scale_len = (extent[1] - extent[0]) * 0.15
        add_scale_bar(ax, scale_len)

    fig.suptitle("Kernel Density Estimation & Overlap", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "08_kde_overlap.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ KDE overlap analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 9.  VORONOI NEIGHBOURHOOD ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def voronoi_analysis(df_a, df_b, name_a, name_b, extent, outdir):
    """
    Voronoi tessellation of combined points; measure cross-type
    neighbour fractions.
    """
    ca, cb = coords(df_a), coords(df_b)
    all_pts = np.vstack([ca, cb])
    labels_true = np.array([0] * len(ca) + [1] * len(cb))
    n = len(all_pts)

    if n < 4:
        print("  ⚠ Too few points for Voronoi analysis; skipping.")
        return {}

    vor = Voronoi(all_pts)

    # For each point, find Voronoi neighbours (ridge pairs)
    neighbours = {i: set() for i in range(n)}
    for (i, j) in vor.ridge_points:
        neighbours[i].add(j)
        neighbours[j].add(i)

    # cross-type neighbour fraction per point
    cross_frac = np.zeros(n)
    for i in range(n):
        if len(neighbours[i]) == 0:
            cross_frac[i] = np.nan
            continue
        my_type = labels_true[i]
        n_cross = sum(1 for j in neighbours[i] if labels_true[j] != my_type)
        cross_frac[i] = n_cross / len(neighbours[i])

    mean_cross_a = np.nanmean(cross_frac[:len(ca)])
    mean_cross_b = np.nanmean(cross_frac[len(ca):])

    # expected cross-type fraction under random labelling
    p_a = len(ca) / n
    expected_cross_a = 1 - p_a   # if random, fraction of B neighbours
    expected_cross_b = p_a

    stats = {
        f"Voronoi mean cross-type frac ({name_a})": mean_cross_a,
        f"Voronoi mean cross-type frac ({name_b})": mean_cross_b,
        f"Voronoi expected cross-frac ({name_a}, random)": expected_cross_a,
        f"Voronoi expected cross-frac ({name_b}, random)": expected_cross_b,
        f"Voronoi segregation index ({name_a})": 1 - mean_cross_a / expected_cross_a if expected_cross_a > 0 else np.nan,
        f"Voronoi segregation index ({name_b})": 1 - mean_cross_b / expected_cross_b if expected_cross_b > 0 else np.nan,
    }

    # figure
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)

    # Voronoi diagram coloured by cross-type fraction
    ax = axes[0]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    # draw Voronoi edges
    for (i, j) in vor.ridge_points:
        ridge_idx = list(vor.ridge_points).index([i, j]) if [i, j] in vor.ridge_points.tolist() else None
    for idx, (i, j) in enumerate(vor.ridge_points):
        vertices = vor.ridge_vertices[idx]
        if -1 in vertices:
            continue
        v0, v1 = vor.vertices[vertices[0]], vor.vertices[vertices[1]]
        # colour: cross-type edge?
        is_cross = labels_true[i] != labels_true[j]
        col = COLOR_BOTH if is_cross else "lightgrey"
        lw = 1.2 if is_cross else 0.4
        alpha = 0.8 if is_cross else 0.3
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], color=col, lw=lw, alpha=alpha)

    sc = ax.scatter(all_pts[:, 0], all_pts[:, 1], c=cross_frac, cmap="RdYlGn",
                    s=25, edgecolors="k", linewidths=0.3, vmin=0, vmax=1,
                    zorder=5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046)
    cbar.set_label("Cross-type neighbour fraction")

    legend_elements = [
        Line2D([0], [0], color=COLOR_BOTH, lw=2, label="Cross-type edge"),
        Line2D([0], [0], color="lightgrey", lw=1, label="Same-type edge"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax.set_title("Voronoi Tessellation")
    scale_len = (extent[1] - extent[0]) * 0.15
    add_scale_bar(ax, scale_len)

    # histogram of cross-type fractions
    ax = axes[1]
    bins_f = np.linspace(0, 1, 21)
    ax.hist(cross_frac[:len(ca)], bins=bins_f, color=COLOR_A, alpha=0.6,
            label=name_a, edgecolor="k", linewidth=0.3)
    ax.hist(cross_frac[len(ca):], bins=bins_f, color=COLOR_B, alpha=0.6,
            label=name_b, edgecolor="k", linewidth=0.3)
    ax.axvline(expected_cross_a, color=COLOR_A, ls="--", lw=1.5,
               label=f"Random expect. ({name_a})")
    ax.axvline(expected_cross_b, color=COLOR_B, ls="--", lw=1.5,
               label=f"Random expect. ({name_b})")
    ax.set_xlabel("Cross-type neighbour fraction")
    ax.set_ylabel("Count")
    ax.set_title("Cross-Type Neighbour Distribution")
    ax.legend(fontsize=8)

    fig.suptitle("Voronoi Neighbourhood Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "09_voronoi.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Voronoi neighbourhood analysis saved.")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ──────────────────────────────────────────────────────────────────────────────
def save_summary(all_stats: dict, outdir: Path):
    """Save a summary CSV and print to console."""
    rows = []
    for section, sdict in all_stats.items():
        for metric, value in sdict.items():
            rows.append({"Section": section, "Metric": metric, "Value": value})
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(outdir / "summary_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("COLOCALIZATION ANALYSIS SUMMARY")
    print("=" * 70)
    for section, sdict in all_stats.items():
        print(f"\n── {section} ──")
        for metric, value in sdict.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
    print("=" * 70)
    print(f"\nAll figures and metrics saved to:  {outdir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_analysis(csv_a: str, csv_b: str, name_a: str = "Population A",
                 name_b: str = "Population B",
                 outdir: str = "colocalisation_results"):
    """Run full colocalization pipeline."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_a} ...")
    df_a = load_csv(csv_a)
    print(f"  → {len(df_a)} objects")
    print(f"Loading {csv_b} ...")
    df_b = load_csv(csv_b)
    print(f"  → {len(df_b)} objects")

    extent = compute_extent(df_a, df_b)
    print(f"ROI extent: x=[{extent[0]:.1f}, {extent[1]:.1f}]  "
          f"y=[{extent[2]:.1f}, {extent[3]:.1f}]\n")

    all_stats = {}

    print("Running analyses...")

    # 1
    plot_spatial_overlay(df_a, df_b, name_a, name_b, extent, outdir)

    # 2
    all_stats["Nearest-Neighbour"] = nearest_neighbour_analysis(
        df_a, df_b, name_a, name_b, extent, outdir
    )

    # 3
    all_stats["Grid Correlation"] = grid_correlation_analysis(
        df_a, df_b, name_a, name_b, extent, outdir
    )

    # 4
    all_stats["Proximity Network"] = proximity_network_analysis(
        df_a, df_b, name_a, name_b, extent, outdir
    )

    # 5
    all_stats["K-Means"] = kmeans_analysis(
        df_a, df_b, name_a, name_b, extent, outdir
    )

    # 6
    all_stats["DBSCAN"] = dbscan_analysis(
        df_a, df_b, name_a, name_b, extent, outdir
    )

    # 7
    all_stats["Ripley's Cross-K"] = ripleys_cross_k(
        df_a, df_b, name_a, name_b, extent, outdir
    )

    # 8
    all_stats["KDE Overlap"] = kde_overlap_analysis(
        df_a, df_b, name_a, name_b, extent, outdir
    )

    # 9
    all_stats["Voronoi"] = voronoi_analysis(
        df_a, df_b, name_a, name_b, extent, outdir
    )

    # summary
    save_summary(all_stats, outdir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive colocalization analysis between two CSV files."
    )
    parser.add_argument("csv_a", help="Path to first CSV file")
    parser.add_argument("csv_b", help="Path to second CSV file")
    parser.add_argument("--name-a", default="Population A",
                        help="Label for first population")
    parser.add_argument("--name-b", default="Population B",
                        help="Label for second population")
    parser.add_argument("-o", "--outdir", default="colocalisation_results",
                        help="Output directory")
    args = parser.parse_args()

    run_analysis(args.csv_a, args.csv_b, args.name_a, args.name_b, args.outdir)