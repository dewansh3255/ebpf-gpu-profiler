#!/usr/bin/env python3
"""
G_21_cpu_gpu_correlation_plot.py
Group 21 - GRS Project Part B

Generates 3 plot variants showing CPU scheduling latency,
context switches, and their effect on GPU utilization —
native vs containerised environments.

Outputs (all to results/plots_hardcoded/):
  12A_cpu_gpu_sidebyside.png  — Option A: side-by-side columns
  12B_cpu_gpu_overlaid.png    — Option B: overlaid on same axes
  12C_cpu_gpu_scatter.png     — Option C: overlay + correlation scatter

Usage:
    python3 G_21_cpu_gpu_correlation_plot.py
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE    = "/home/gpu1/eGPU/21_ebpf_egpu/results"
OUT_DIR = "/home/gpu1/eGPU/21_ebpf_egpu/results/plots_hardcoded"

# ─── Palette ──────────────────────────────────────────────────────────────────
NC   = "#2196F3"    # native  – blue
CC   = "#FF5722"    # container – orange-red
NL   = "#BBDEFB"    # native  light fill
CL   = "#FFCCBC"    # container light fill
WIN  = "#E8F5E9"    # active-training-window shading (light green)
WIN2 = "#FFF9C4"    # container active window (light yellow)

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_gpu(env):
    """
    Returns:
        bins  – DataFrame[t_bin(int), gpu_util(float)]  1-second averages
        t0    – float, Unix epoch of first GPU sample (reference t=0)
        act   – (active_start_s, active_end_s) for the training window
    """
    path = os.path.join(BASE, env, "21_gpu_results.csv")
    df = pd.read_csv(path)
    df = df[df["gpu_index"] == 0].copy()
    t0 = df["timestamp"].min()
    df["t"] = df["timestamp"] - t0
    df["t_bin"] = df["t"].apply(lambda x: int(np.floor(x)))
    bins = df.groupby("t_bin")["gpu_util_pct"].mean().reset_index()
    bins.columns = ["t_bin", "gpu_util"]
    active = df[df["gpu_util_pct"] > 5]
    act = (active["t"].min(), active["t"].max()) if len(active) else (0, 0)
    return bins, t0, act


def load_sched(env, t0_unix):
    """
    Reads sched_latency CSV in chunks, bins into 1-second windows.
    Returns DataFrame[t_bin, sched_mean_us, sched_p95_us]
    Timestamps in file are Unix nanoseconds — same reference as GPU.
    """
    path = os.path.join(BASE, env, "21_cpu_results_sched_latency.csv")
    print(f"  [sched {env}] reading …", flush=True)
    acc = defaultdict(list)
    for chunk in pd.read_csv(path, chunksize=100_000,
                             usecols=["timestamp_ns", "latency_ns"]):
        chunk["t_bin"] = ((chunk["timestamp_ns"] / 1e9) - t0_unix).apply(
            lambda x: int(np.floor(x))
        )
        chunk["us"] = chunk["latency_ns"] / 1_000.0
        for b, grp in chunk.groupby("t_bin"):
            acc[b].extend(grp["us"].tolist())

    rows = []
    for b in sorted(acc):
        v = np.array(acc[b])
        rows.append({"t_bin": b,
                     "sched_mean": float(np.mean(v)),
                     "sched_p95":  float(np.percentile(v, 95))})
    return pd.DataFrame(rows)


def load_ctx(env, sched_min_ns, ctx_min_ktime_ns, t0_unix):
    """
    Reads ctx_switches CSV in chunks.
    ctx timestamps are bpf_ktime_get_ns() — nanoseconds since boot.
    Uses sched_min_ns − ctx_min_ktime_ns to convert to Unix nanoseconds.
    Returns DataFrame[t_bin, ctx_per_sec]
    """
    path = os.path.join(BASE, env, "21_cpu_results_ctx_switches.csv")
    print(f"  [ctx   {env}] reading …", flush=True)
    offset_ns = int(sched_min_ns) - int(ctx_min_ktime_ns)
    acc = defaultdict(int)
    for chunk in pd.read_csv(path, chunksize=200_000,
                             usecols=["timestamp_ns"]):
        unix_sec = (chunk["timestamp_ns"] + offset_ns) / 1e9
        t_bins   = (unix_sec - t0_unix).apply(lambda x: int(np.floor(x)))
        for bv, cnt in t_bins.value_counts().items():
            acc[bv] += int(cnt)

    df = pd.DataFrame(list(acc.items()), columns=["t_bin", "ctx_per_sec"])
    return df.sort_values("t_bin").reset_index(drop=True)


def load_all(env):
    print(f"\n[{env.upper()}] Loading data …", flush=True)
    gpu, t0, act = load_gpu(env)

    sp = os.path.join(BASE, env, "21_cpu_results_sched_latency.csv")
    cp = os.path.join(BASE, env, "21_cpu_results_ctx_switches.csv")
    sched_min = pd.read_csv(sp, usecols=["timestamp_ns"],
                            nrows=1000)["timestamp_ns"].min()
    ctx_min   = pd.read_csv(cp, usecols=["timestamp_ns"],
                            nrows=1000)["timestamp_ns"].min()

    sched = load_sched(env, t0)
    ctx   = load_ctx(env, sched_min, ctx_min, t0)
    return gpu, sched, ctx, act


def smooth(series, w=3):
    return series.rolling(w, center=True, min_periods=1).mean()


# ─── SHARED AXIS LIMITS ────────────────────────────────────────────────────────

def get_ylims(n_sched, c_sched, n_ctx, c_ctx):
    """Compute shared Y-axis limits for both environments."""
    sched_max = max(n_sched["sched_p95"].quantile(0.97),
                    c_sched["sched_p95"].quantile(0.97))
    sched_max = min(sched_max, 250)   # cap at 250 μs for readability
    ctx_max   = max(n_ctx["ctx_per_sec"].quantile(0.97),
                    c_ctx["ctx_per_sec"].quantile(0.97)) * 1.15
    return sched_max, ctx_max


# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────

def _shade_active(ax, act, color=WIN, alpha=0.25, label=None):
    ax.axvspan(act[0], act[1], color=color, alpha=alpha, zorder=0, label=label)


def _draw_gpu(ax, gpu, color, label, act=None, act_color=WIN, alpha=0.85):
    if act:
        _shade_active(ax, act, color=act_color, alpha=0.18)
    ax.fill_between(gpu["t_bin"], 0, smooth(gpu["gpu_util"]),
                    color=color, alpha=0.25)
    ax.plot(gpu["t_bin"], smooth(gpu["gpu_util"]),
            color=color, lw=2, label=label)
    ax.set_ylim(0, 105)
    ax.set_ylabel("GPU 0 Util (%)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))


def _draw_sched(ax, sched, color, light, label, sched_max):
    ax.fill_between(sched["t_bin"],
                    smooth(sched["sched_mean"]),
                    smooth(sched["sched_p95"]),
                    color=light, alpha=0.55, label="P95 band")
    ax.plot(sched["t_bin"], smooth(sched["sched_mean"]),
            color=color, lw=2, label=label)
    ax.set_ylim(0, sched_max * 1.15)
    ax.set_ylabel("Sched Latency (μs)", fontsize=11)


def _draw_ctx(ax, ctx, color, label, ctx_max):
    ax.fill_between(ctx["t_bin"], 0, smooth(ctx["ctx_per_sec"]),
                    color=color, alpha=0.22)
    ax.plot(ctx["t_bin"], smooth(ctx["ctx_per_sec"]),
            color=color, lw=2, label=label)
    ax.set_ylim(0, ctx_max)
    ax.set_ylabel("Ctx Switches / sec", fontsize=11)


def _annotate(ax, text, x=0.02, y=0.96, fontsize=9):
    ax.text(x, y, text, transform=ax.transAxes,
            va="top", ha="left", fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.75, edgecolor="#cccccc"))


def _fmt_stats(gpu, sched, ctx, act):
    """Build annotation string from active-window averages."""
    mask_gpu   = (gpu["t_bin"] >= act[0]) & (gpu["t_bin"] <= act[1])
    mask_sched = (sched["t_bin"] >= act[0]) & (sched["t_bin"] <= act[1])
    mask_ctx   = (ctx["t_bin"]  >= act[0]) & (ctx["t_bin"]  <= act[1])

    util_avg  = gpu.loc[mask_gpu,   "gpu_util"].mean()
    sched_avg = sched.loc[mask_sched, "sched_mean"].mean()
    ctx_avg   = ctx.loc[mask_ctx,   "ctx_per_sec"].mean()

    return (f"Active window\n"
            f"GPU util : {util_avg:.1f}%\n"
            f"Sched mean: {sched_avg:.1f} μs\n"
            f"Ctx/s    : {ctx_avg:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION A — SIDE-BY-SIDE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_option_a(n_gpu, n_sched, n_ctx, n_act,
                  c_gpu, c_sched, c_ctx, c_act,
                  sched_max, ctx_max):
    print("\n[Plot A] Side-by-side …", flush=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12),
                              sharex="col", sharey="row")
    fig.subplots_adjust(hspace=0.10, wspace=0.22,
                        top=0.92, bottom=0.07, left=0.08, right=0.97)

    fig.suptitle(
        "CPU Scheduler Overhead & Context Switches → GPU Impact\n"
        "Option A — Native (left) vs Container (right)",
        fontsize=14, fontweight="bold"
    )

    cols = [(n_gpu, n_sched, n_ctx, n_act, NC, NL, "Native"),
            (c_gpu, c_sched, c_ctx, c_act, CC, CL, "Container")]

    for col_i, (gpu, sched, ctx, act, color, light, label) in enumerate(cols):
        ax_gpu   = axes[0][col_i]
        ax_sched = axes[1][col_i]
        ax_ctx   = axes[2][col_i]

        # GPU util
        _shade_active(ax_gpu, act, color=WIN if col_i == 0 else WIN2, alpha=0.22)
        _draw_gpu(ax_gpu, gpu, color, label)
        ax_gpu.set_title(label, fontsize=13, fontweight="bold", color=color)

        # Sched latency
        _shade_active(ax_sched, act, color=WIN if col_i == 0 else WIN2, alpha=0.22)
        _draw_sched(ax_sched, sched, color, light, "Mean latency", sched_max)
        # P95 legend entry
        ax_sched.plot([], [], color=light, lw=8, alpha=0.7, label="P95 band")

        # Ctx switches
        _shade_active(ax_ctx, act, color=WIN if col_i == 0 else WIN2, alpha=0.22)
        _draw_ctx(ax_ctx, ctx, color, "Ctx switches/s", ctx_max)
        ax_ctx.set_xlabel("Time (seconds)", fontsize=11)

        # Stats annotation (top-right corner of GPU panel)
        _annotate(ax_gpu, _fmt_stats(gpu, sched, ctx, act), x=0.97, y=0.97)
        ax_gpu.xaxis.set_visible(False)

        # Legends
        for ax in [ax_gpu, ax_sched, ax_ctx]:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            ax.grid(True, alpha=0.25, axis="y")
            ax.axvspan(act[0], act[1], color=color, alpha=0.04, zorder=0)

    # shared Y labels on left only
    axes[0][1].set_ylabel("")
    axes[1][1].set_ylabel("")
    axes[2][1].set_ylabel("")

    # Active-window legend patch
    n_patch = mpatches.Patch(color=WIN,  alpha=0.5, label="Native active window")
    c_patch = mpatches.Patch(color=WIN2, alpha=0.5, label="Container active window")
    fig.legend(handles=[n_patch, c_patch], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    out = os.path.join(OUT_DIR, "12A_cpu_gpu_sidebyside.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION B — OVERLAID
# ═══════════════════════════════════════════════════════════════════════════════

def plot_option_b(n_gpu, n_sched, n_ctx, n_act,
                  c_gpu, c_sched, c_ctx, c_act,
                  sched_max, ctx_max):
    print("\n[Plot B] Overlaid …", flush=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 13),
                              sharex=False)
    fig.subplots_adjust(hspace=0.28, top=0.91, bottom=0.07,
                        left=0.10, right=0.96)

    fig.suptitle(
        "CPU Scheduler Overhead & Context Switches → GPU Impact\n"
        "Option B — Native vs Container Overlaid (each run normalised to t = 0)",
        fontsize=13, fontweight="bold"
    )

    ax_gpu, ax_sched, ax_ctx = axes

    # ── GPU util ──────────────────────────────────────────────────────────────
    _shade_active(ax_gpu, n_act, color=NC, alpha=0.10, label="Native active")
    _shade_active(ax_gpu, c_act, color=CC, alpha=0.10, label="Container active")
    ax_gpu.fill_between(n_gpu["t_bin"], 0, smooth(n_gpu["gpu_util"]),
                        color=NC, alpha=0.18)
    ax_gpu.fill_between(c_gpu["t_bin"], 0, smooth(c_gpu["gpu_util"]),
                        color=CC, alpha=0.18)
    ax_gpu.plot(n_gpu["t_bin"], smooth(n_gpu["gpu_util"]),
                color=NC, lw=2.5, label=f"Native  (mean active={n_gpu[n_gpu['t_bin'].between(*n_act)]['gpu_util'].mean():.1f}%)")
    ax_gpu.plot(c_gpu["t_bin"], smooth(c_gpu["gpu_util"]),
                color=CC, lw=2.5, label=f"Container  (mean active={c_gpu[c_gpu['t_bin'].between(*c_act)]['gpu_util'].mean():.1f}%)")
    ax_gpu.set_ylim(0, 105)
    ax_gpu.set_ylabel("GPU 0 Util (%)", fontsize=11)
    ax_gpu.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax_gpu.set_title("① GPU Utilisation (GPU 0)", fontsize=11, fontweight="bold")
    ax_gpu.grid(True, alpha=0.25, axis="y")
    ax_gpu.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # ── Sched latency ─────────────────────────────────────────────────────────
    ax_sched.fill_between(n_sched["t_bin"],
                          smooth(n_sched["sched_mean"]),
                          smooth(n_sched["sched_p95"]),
                          color=NL, alpha=0.60, label="Native P95 band")
    ax_sched.fill_between(c_sched["t_bin"],
                          smooth(c_sched["sched_mean"]),
                          smooth(c_sched["sched_p95"]),
                          color=CL, alpha=0.55, label="Container P95 band")
    n_smean = n_sched.loc[n_sched["t_bin"].between(*n_act), "sched_mean"].mean()
    c_smean = c_sched.loc[c_sched["t_bin"].between(*c_act), "sched_mean"].mean()
    ax_sched.plot(n_sched["t_bin"], smooth(n_sched["sched_mean"]),
                  color=NC, lw=2.5, label=f"Native mean  ({n_smean:.1f} μs)")
    ax_sched.plot(c_sched["t_bin"], smooth(c_sched["sched_mean"]),
                  color=CC, lw=2.5, label=f"Container mean  ({c_smean:.1f} μs)")
    ax_sched.set_ylim(0, sched_max * 1.15)
    ax_sched.set_ylabel("Sched Latency (μs)", fontsize=11)
    ax_sched.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax_sched.set_title("② CPU Scheduler Wakeup Latency (system-wide, 1 s bins)",
                       fontsize=11, fontweight="bold")
    ax_sched.grid(True, alpha=0.25, axis="y")

    # ── Ctx switches ──────────────────────────────────────────────────────────
    ax_ctx.fill_between(n_ctx["t_bin"], 0, smooth(n_ctx["ctx_per_sec"]),
                        color=NC, alpha=0.18)
    ax_ctx.fill_between(c_ctx["t_bin"], 0, smooth(c_ctx["ctx_per_sec"]),
                        color=CC, alpha=0.18)
    n_crate = n_ctx.loc[n_ctx["t_bin"].between(*n_act), "ctx_per_sec"].mean()
    c_crate = c_ctx.loc[c_ctx["t_bin"].between(*c_act), "ctx_per_sec"].mean()
    ax_ctx.plot(n_ctx["t_bin"], smooth(n_ctx["ctx_per_sec"]),
                color=NC, lw=2.5, label=f"Native  ({n_crate:,.0f}/s avg)")
    ax_ctx.plot(c_ctx["t_bin"], smooth(c_ctx["ctx_per_sec"]),
                color=CC, lw=2.5, label=f"Container  ({c_crate:,.0f}/s avg)")
    ax_ctx.set_ylim(0, ctx_max)
    ax_ctx.set_ylabel("Ctx Switches / sec", fontsize=11)
    ax_ctx.set_xlabel("Time (seconds from start of monitoring)", fontsize=11)
    ax_ctx.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax_ctx.set_title("③ Context Switch Rate (system-wide, 1 s bins)",
                     fontsize=11, fontweight="bold")
    ax_ctx.grid(True, alpha=0.25, axis="y")

    # Key insight annotation
    fig.text(0.01, 0.02,
             "Active training window shaded above (GPU util > 5%)  |  "
             "Sched latency = time between sched_wakeup and sched_switch  |  "
             "3-second rolling average applied to all lines",
             fontsize=8, color="#555555")

    out = os.path.join(OUT_DIR, "12B_cpu_gpu_overlaid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION C — OVERLAID + SCATTER CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_option_c(n_gpu, n_sched, n_ctx, n_act,
                  c_gpu, c_sched, c_ctx, c_act,
                  sched_max, ctx_max):
    print("\n[Plot C] Overlay + scatter …", flush=True)
    fig = plt.figure(figsize=(14, 17))
    gs  = fig.add_gridspec(4, 2,
                           height_ratios=[1, 1, 1, 1.2],
                           hspace=0.45, wspace=0.30,
                           top=0.90, bottom=0.06,
                           left=0.09, right=0.97)

    fig.suptitle(
        "CPU Scheduler Overhead & Context Switches → GPU Impact\n"
        "Option C — Overlay Time-Series + Correlation Scatter",
        fontsize=13, fontweight="bold"
    )

    ax_gpu   = fig.add_subplot(gs[0, :])
    ax_sched = fig.add_subplot(gs[1, :])
    ax_ctx   = fig.add_subplot(gs[2, :])
    ax_sc1   = fig.add_subplot(gs[3, 0])
    ax_sc2   = fig.add_subplot(gs[3, 1])

    # Shared X-axis label explaining normalisation — added to each time-series panel
    XAXIS_LABEL = (
        "Elapsed time within each independent run (seconds from t = 0)\n"
        "[ Native run and Container run were recorded separately — "
        "both are shifted to t = 0 for side-by-side comparison ]"
    )

    # ── Rows 1-3 ─────────────────────────────────────────────────────────────
    for (gpu, sched, ctx, act, color, light, label) in [
        (n_gpu, n_sched, n_ctx, n_act, NC, NL, "Native"),
        (c_gpu, c_sched, c_ctx, c_act, CC, CL, "Container"),
    ]:
        ax_gpu.fill_between(gpu["t_bin"], 0, smooth(gpu["gpu_util"]),
                            color=color, alpha=0.16)
        ax_gpu.plot(gpu["t_bin"], smooth(gpu["gpu_util"]),
                    color=color, lw=2.2, label=label)

        ax_sched.fill_between(sched["t_bin"],
                              smooth(sched["sched_mean"]),
                              smooth(sched["sched_p95"]),
                              color=light, alpha=0.50)
        ax_sched.plot(sched["t_bin"], smooth(sched["sched_mean"]),
                      color=color, lw=2.2, label=label)

        ax_ctx.fill_between(ctx["t_bin"], 0, smooth(ctx["ctx_per_sec"]),
                            color=color, alpha=0.16)
        ax_ctx.plot(ctx["t_bin"], smooth(ctx["ctx_per_sec"]),
                    color=color, lw=2.2, label=label)

        # Shade active training window + vertical markers with labels
        for ax in [ax_gpu, ax_sched, ax_ctx]:
            ax.axvspan(act[0], act[1], color=color, alpha=0.10, zorder=0)
            ax.axvline(act[0], color=color, lw=1.2, ls=":", alpha=0.7)
            ax.axvline(act[1], color=color, lw=1.2, ls=":", alpha=0.7)

        # Label the training window on the GPU panel only (keeps it uncluttered)
        ypos = 102 if color == NC else 95
        ax_gpu.annotate(
            f"{label}\ntraining\n{act[0]:.0f}s – {act[1]:.0f}s",
            xy=((act[0] + act[1]) / 2, ypos),
            ha="center", va="top", fontsize=7.5, color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                      alpha=0.85, lw=0.8),
        )

    ax_gpu.set_ylim(0, 110)
    ax_gpu.set_ylabel("GPU 0 Util (%)", fontsize=10)
    ax_gpu.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax_gpu.set_title(
        "① GPU Utilisation  —  two separate runs, both time-shifted to t = 0",
        fontsize=11, fontweight="bold"
    )
    ax_gpu.grid(True, alpha=0.25, axis="y")
    ax_gpu.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_gpu.set_xlabel(XAXIS_LABEL, fontsize=8.5, color="#444444", labelpad=6)

    # Prominent note box inside the GPU panel
    ax_gpu.text(
        0.99, 0.97,
        "⚠  These runs did NOT happen simultaneously.\n"
        "    Native was run first, then Container separately.\n"
        "    Both X-axes are independently shifted to t = 0\n"
        "    so the profiles can be compared directly.",
        transform=ax_gpu.transAxes,
        va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4",
                  edgecolor="#F9A825", alpha=0.95, lw=1.2),
    )

    ax_sched.set_ylim(0, sched_max * 1.15)
    ax_sched.set_ylabel("Sched Latency (μs)", fontsize=10)
    ax_sched.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax_sched.set_title("② CPU Scheduler Wakeup Latency  (mean + P95 band, 1 s bins)",
                       fontsize=11, fontweight="bold")
    ax_sched.grid(True, alpha=0.25, axis="y")
    ax_sched.set_xlabel(XAXIS_LABEL, fontsize=8.5, color="#444444", labelpad=6)

    ax_ctx.set_ylim(0, ctx_max)
    ax_ctx.set_ylabel("Ctx Switches / sec", fontsize=10)
    ax_ctx.set_xlabel(XAXIS_LABEL, fontsize=8.5, color="#444444", labelpad=6)
    ax_ctx.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax_ctx.set_title("③ Context Switch Rate  (1 s bins)", fontsize=11, fontweight="bold")
    ax_ctx.grid(True, alpha=0.25, axis="y")

    # ── Scatter 1: Sched latency vs GPU util ─────────────────────────────────
    def merge_active(gpu, sched, act):
        """Join GPU util and sched_mean on t_bin within the active window."""
        merged = pd.merge(gpu, sched, on="t_bin")
        return merged[merged["t_bin"].between(act[0], act[1])]

    n_merged = merge_active(n_gpu, n_sched, n_act)
    c_merged = merge_active(c_gpu, c_sched, c_act)

    ax_sc1.scatter(n_merged["sched_mean"], n_merged["gpu_util"],
                   color=NC, alpha=0.65, s=45, label="Native", edgecolors="white", lw=0.4)
    ax_sc1.scatter(c_merged["sched_mean"], c_merged["gpu_util"],
                   color=CC, alpha=0.65, s=45, label="Container", edgecolors="white", lw=0.4)

    # trend lines
    for data, color in [(n_merged, NC), (c_merged, CC)]:
        if len(data) > 3:
            z = np.polyfit(data["sched_mean"], data["gpu_util"], 1)
            x_line = np.linspace(data["sched_mean"].min(),
                                 data["sched_mean"].max(), 50)
            ax_sc1.plot(x_line, np.poly1d(z)(x_line),
                        color=color, lw=1.5, ls="--", alpha=0.8)

    ax_sc1.set_xlabel("Mean Sched Latency / sec bin (μs)", fontsize=10)
    ax_sc1.set_ylabel("GPU 0 Util (%)", fontsize=10)
    ax_sc1.set_title("④ Scatter: Sched Latency → GPU Util\n(active training, 1 pt = 1 second)",
                     fontsize=10, fontweight="bold")
    ax_sc1.legend(fontsize=9, framealpha=0.8)
    ax_sc1.grid(True, alpha=0.3)
    ax_sc1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # ── Scatter 2: Ctx switches vs GPU util ──────────────────────────────────
    def merge_ctx_active(gpu, ctx, act):
        merged = pd.merge(gpu, ctx, on="t_bin")
        return merged[merged["t_bin"].between(act[0], act[1])]

    n_mc = merge_ctx_active(n_gpu, n_ctx, n_act)
    c_mc = merge_ctx_active(c_gpu, c_ctx, c_act)

    ax_sc2.scatter(n_mc["ctx_per_sec"] / 1000, n_mc["gpu_util"],
                   color=NC, alpha=0.65, s=45, label="Native", edgecolors="white", lw=0.4)
    ax_sc2.scatter(c_mc["ctx_per_sec"] / 1000, c_mc["gpu_util"],
                   color=CC, alpha=0.65, s=45, label="Container", edgecolors="white", lw=0.4)

    for data, color in [(n_mc, NC), (c_mc, CC)]:
        if len(data) > 3:
            z = np.polyfit(data["ctx_per_sec"], data["gpu_util"], 1)
            x_line = np.linspace(data["ctx_per_sec"].min(),
                                 data["ctx_per_sec"].max(), 50)
            ax_sc2.plot(x_line / 1000, np.poly1d(z)(x_line),
                        color=color, lw=1.5, ls="--", alpha=0.8)

    ax_sc2.set_xlabel("Context Switches / sec (thousands)", fontsize=10)
    ax_sc2.set_ylabel("GPU 0 Util (%)", fontsize=10)
    ax_sc2.set_title("⑤ Scatter: Ctx Switch Rate → GPU Util\n(active training, 1 pt = 1 second)",
                     fontsize=10, fontweight="bold")
    ax_sc2.legend(fontsize=9, framealpha=0.8)
    ax_sc2.grid(True, alpha=0.3)
    ax_sc2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    fig.text(0.01, 0.015,
             "Sched latency = eBPF sched:sched_wakeup → sched:sched_switch  |  "
             "Ctx switches = eBPF sched:sched_switch events  |  "
             "Scatter dashed = linear trend  |  3 s rolling avg on time-series",
             fontsize=8, color="#555555")

    out = os.path.join(OUT_DIR, "12C_cpu_gpu_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    n_gpu, n_sched, n_ctx, n_act = load_all("native")
    c_gpu, c_sched, c_ctx, c_act = load_all("container")

    sched_max, ctx_max = get_ylims(n_sched, c_sched, n_ctx, c_ctx)
    print(f"\n  Shared limits → sched_max={sched_max:.1f} μs, "
          f"ctx_max={ctx_max:,.0f}/s")

    plot_option_a(n_gpu, n_sched, n_ctx, n_act,
                  c_gpu, c_sched, c_ctx, c_act, sched_max, ctx_max)

    plot_option_b(n_gpu, n_sched, n_ctx, n_act,
                  c_gpu, c_sched, c_ctx, c_act, sched_max, ctx_max)

    plot_option_c(n_gpu, n_sched, n_ctx, n_act,
                  c_gpu, c_sched, c_ctx, c_act, sched_max, ctx_max)

    print("\nAll 3 plots saved to results/plots_hardcoded/")
