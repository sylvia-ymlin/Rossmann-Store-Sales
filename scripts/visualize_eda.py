"""
EDA visualizations for the Rossmann Store Sales dataset.
Generates 4 charts and saves them to assets/.

Usage:
    python scripts/visualize_eda.py
"""

# ruff: noqa: E402
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F8F8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def load_data() -> pd.DataFrame:
    print(f"Loading {TRAIN_PATH} ...")
    df = pd.read_csv(TRAIN_PATH, low_memory=False, parse_dates=["Date"])
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    print(f"  {len(df):,} rows after filtering closed/zero-sales records")
    return df


# ── Plot 1: Monthly sales trend ───────────────────────────────────────────────
def plot_sales_trend(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby(df["Date"].dt.to_period("M"))["Sales"]
        .mean()
        .reset_index()
    )
    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["Date"], monthly["Sales"], color=PALETTE[0], linewidth=2)
    ax.fill_between(monthly["Date"], monthly["Sales"], alpha=0.15, color=PALETTE[0])
    ax.set_title("Average Daily Sales by Month (2013 – 2015)", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Avg Sales (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    out = ASSETS_DIR / "sales_trend.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 2: Sales by day of week ──────────────────────────────────────────────
def plot_sales_by_dow(df: pd.DataFrame) -> None:
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # DayOfWeek in dataset: 1=Mon … 7=Sun
    groups = [df.loc[df["DayOfWeek"] == d, "Sales"].values for d in range(1, 8)]

    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot(
        groups,
        tick_labels=dow_labels,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], [PALETTE[0]] * 5 + [PALETTE[1]] + [PALETTE[2]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_title("Sales Distribution by Day of Week", fontsize=13, pad=12)
    ax.set_ylabel("Sales (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    out = ASSETS_DIR / "sales_by_dayofweek.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 3: Promo effect ──────────────────────────────────────────────────────
def plot_promo_effect(df: pd.DataFrame) -> None:
    promo_mean = df.groupby(["DayOfWeek", "Promo"])["Sales"].mean().reset_index()
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4))
    for promo_val, label, color, offset in [
        (0, "No Promo", PALETTE[0], -width / 2),
        (1, "Promo",    PALETTE[1],  width / 2),
    ]:
        vals = [
            promo_mean.loc[
                (promo_mean["DayOfWeek"] == d) & (promo_mean["Promo"] == promo_val),
                "Sales",
            ].values[0] if len(promo_mean.loc[
                (promo_mean["DayOfWeek"] == d) & (promo_mean["Promo"] == promo_val)
            ]) else 0
            for d in range(1, 8)
        ]
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(dow_labels)
    ax.set_title("Average Sales: Promo vs No Promo by Day of Week", fontsize=13, pad=12)
    ax.set_ylabel("Avg Sales (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    out = ASSETS_DIR / "promo_effect.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 4: Sales distribution + log transform ────────────────────────────────
def plot_sales_distribution(df: pd.DataFrame) -> None:
    sample = df["Sales"].sample(n=min(100_000, len(df)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(sample, bins=80, color=PALETTE[0], alpha=0.85, edgecolor="none")
    axes[0].set_title("Sales Distribution (raw)", fontsize=12)
    axes[0].set_xlabel("Sales (€)")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    axes[1].hist(np.log1p(sample), bins=80, color=PALETTE[2], alpha=0.85, edgecolor="none")
    axes[1].set_title("Sales Distribution (log1p)", fontsize=12)
    axes[1].set_xlabel("log1p(Sales)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Target Variable Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    out = ASSETS_DIR / "sales_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main() -> None:
    df = load_data()
    print("Generating plots...")
    plot_sales_trend(df)
    plot_sales_by_dow(df)
    plot_promo_effect(df)
    plot_sales_distribution(df)
    print("Done.")


if __name__ == "__main__":
    main()
