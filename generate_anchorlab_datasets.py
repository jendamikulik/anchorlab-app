#!/usr/bin/env python3
"""
generate_anchorlab_datasets.py

Generate synthetic CSV datasets for AnchorLab testing.

Output format:
    x,y

Each dataset is designed to stress a different part of AnchorLab:
- Gaussian hammer / smoothing
- barycenter stability
- closed-loop anchoring
- Fourier-sector decomposition

Usage examples:
    python generate_anchorlab_datasets.py
    python generate_anchorlab_datasets.py --out ./datasets
    python generate_anchorlab_datasets.py --n 2048 --seed 123 --out ./datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Basic utilities
# ============================================================

def gaussian(x: np.ndarray, mu: float, sigma: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def hermite_phys(n: int, z: np.ndarray) -> np.ndarray:
    """
    Physicists' Hermite polynomial H_n(z), built by recurrence.
    """
    if n == 0:
        return np.ones_like(z)
    if n == 1:
        return 2.0 * z
    h_nm2 = np.ones_like(z)
    h_nm1 = 2.0 * z
    for k in range(1, n):
        h_n = 2.0 * z * h_nm1 - 2.0 * k * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def hermite_gaussian(x: np.ndarray, n: int, center: float = 0.0, scale: float = 1.0, amp: float = 1.0) -> np.ndarray:
    """
    Unnormalized Hermite-Gaussian mode:
        H_n((x-center)/scale) * exp(-((x-center)/scale)^2 / 2)
    """
    z = (x - center) / scale
    return amp * hermite_phys(n, z) * np.exp(-0.5 * z**2)


def chirp_signal(x: np.ndarray, x0: float, width: float, f0: float, f1: float, amp: float = 1.0) -> np.ndarray:
    """
    Windowed chirp: frequency sweeps from f0 to f1 over the support.
    """
    z = (x - x0) / width
    window = np.exp(-0.5 * z**2)
    phase = 2.0 * np.pi * (f0 * x + 0.5 * (f1 - f0) * (x - x.min())**2 / (x.max() - x.min()))
    return amp * window * np.sin(phase)


def add_noise(y: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    return y + sigma * rng.normal(size=len(y))


def normalize_peak(y: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = np.max(np.abs(y))
    if m <= 1e-15:
        return y.copy()
    return peak * y / m


def save_xy(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(path, index=False)


# ============================================================
# Dataset builders
# ============================================================

def ds_spike_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = 0.15 * rng.normal(size=len(x))
    y += gaussian(x, mu=5.0, sigma=0.035, amp=3.0)
    y += 0.18 * np.sin(5.5 * x)
    return normalize_peak(y)


def ds_two_peaks_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = 0.10 * rng.normal(size=len(x))
    y += gaussian(x, mu=3.0, sigma=0.18, amp=1.8)
    y += gaussian(x, mu=7.15, sigma=0.35, amp=1.2)
    return normalize_peak(y)


def ds_step_impulse(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.where(x > 4.2, 0.8, -0.1)
    y += gaussian(x, mu=6.75, sigma=0.03, amp=2.8)
    y = add_noise(y, rng, sigma=0.05)
    return normalize_peak(y)


def ds_chirp_burst(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = chirp_signal(x, x0=5.0, width=1.5, f0=0.7, f1=8.0, amp=1.0)
    y += 0.2 * np.sin(2.0 * np.pi * 0.35 * x)
    y = add_noise(y, rng, sigma=0.04)
    return normalize_peak(y)


def ds_self_dualish_gaussian(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Not exactly Fourier-fixed in the discrete app, but behaves very cleanly.
    y = gaussian(x, mu=5.0, sigma=0.55, amp=1.0)
    y += 0.02 * rng.normal(size=len(x))
    return normalize_peak(y)


def ds_odd_even_mix(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Mixture of Hermite-Gaussian-like modes centered in the domain.
    Useful for seeing structured sector content.
    """
    c = 5.0
    s = 0.7
    y = (
        1.0 * hermite_gaussian(x, 0, center=c, scale=s)
        + 0.7 * hermite_gaussian(x, 1, center=c, scale=s)
        - 0.45 * hermite_gaussian(x, 2, center=c, scale=s)
        + 0.25 * hermite_gaussian(x, 3, center=c, scale=s)
    )
    y = add_noise(y, rng, sigma=0.015)
    return normalize_peak(y)


def ds_offcenter_triplet(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Good for barycenter drift / anchor tests.
    """
    y = (
        0.9 * gaussian(x, mu=2.0, sigma=0.16)
        + 1.4 * gaussian(x, mu=5.4, sigma=0.22)
        + 0.7 * gaussian(x, mu=8.1, sigma=0.14)
    )
    y += 0.08 * np.sin(10.0 * x)
    y = add_noise(y, rng, sigma=0.03)
    return normalize_peak(y)


def ds_sector_torture(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Deliberately ugly signal: impulses + chirp + oscillation + broad bump.
    This is the 'echt masakr' stress test.
    """
    y = (
        2.2 * gaussian(x, mu=1.6, sigma=0.02)
        - 1.7 * gaussian(x, mu=4.9, sigma=0.03)
        + 1.2 * gaussian(x, mu=8.2, sigma=0.025)
        + 0.65 * gaussian(x, mu=6.0, sigma=0.8)
    )
    y += chirp_signal(x, x0=5.3, width=1.2, f0=1.5, f1=12.0, amp=0.8)
    y += 0.25 * np.sin(18.0 * x + 0.4)
    y = add_noise(y, rng, sigma=0.08)
    return normalize_peak(y)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CSV datasets for AnchorLab.")
    parser.add_argument("--out", type=str, default="anchorlab_datasets", help="Output directory")
    parser.add_argument("--n", type=int, default=1200, help="Number of samples")
    parser.add_argument("--xmin", type=float, default=0.0, help="Grid minimum")
    parser.add_argument("--xmax", type=float, default=10.0, help="Grid maximum")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    x = np.linspace(args.xmin, args.xmax, args.n)

    datasets = {
        "01_spike_noise.csv": ds_spike_noise(x, rng),
        "02_two_peaks_noise.csv": ds_two_peaks_noise(x, rng),
        "03_step_impulse.csv": ds_step_impulse(x, rng),
        "04_chirp_burst.csv": ds_chirp_burst(x, rng),
        "05_self_dualish_gaussian.csv": ds_self_dualish_gaussian(x, rng),
        "06_odd_even_mix.csv": ds_odd_even_mix(x, rng),
        "07_offcenter_triplet.csv": ds_offcenter_triplet(x, rng),
        "08_sector_torture.csv": ds_sector_torture(x, rng),
    }

    summary_rows = []
    for name, y in datasets.items():
        path = out_dir / name
        save_xy(path, x, y)
        summary_rows.append(
            {
                "file": name,
                "samples": len(y),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "mean_abs": float(np.mean(np.abs(y))),
                "std": float(np.std(y)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)

    print("=" * 72)
    print("AnchorLab dataset pack generated")
    print("=" * 72)
    print(f"Output directory: {out_dir.resolve()}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Suggested use:")
    print("  - 01,02,03,07  -> hammer / barycenter / anchor tests")
    print("  - 04,06,08     -> Fourier-sector tests")
    print("  - 05           -> smooth self-dualish baseline")
    print("=" * 72)


if __name__ == "__main__":
    main()
