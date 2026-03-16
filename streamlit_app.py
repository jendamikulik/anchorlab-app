import io
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# AnchorLab MVP+
# ------------------------------------------------------------
# Raw signal -> Gaussian hammer -> stable readout -> anchor
# plus
# Discrete Fourier sector decomposition:
#   f = P_1 f + P_{-i} f + P_{-1} f + P_i f
# where
#   P_lambda = (1/4) * sum_{m=0}^3 lambda^{-m} F^m
# for the unitary discrete Fourier operator F.
# ------------------------------------------------------------
# Run:
#   streamlit run anchorlab_app_v2.py
# ============================================================


st.set_page_config(page_title="AnchorLab", layout="wide")


@dataclass
class LoopParams:
    alpha: float
    beta: float
    x_star: float
    lam: float
    sigma: float


FOURIER_EIGENVALUES: dict[str, complex] = {
    "1": 1.0 + 0.0j,
    "-i": -1.0j,
    "-1": -1.0 + 0.0j,
    "i": 1.0j,
}


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Return a normalized discrete 1D Gaussian kernel."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if radius is None:
        radius = max(3, int(math.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_hammer(y: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a signal with a Gaussian kernel."""
    kernel = gaussian_kernel_1d(sigma)
    return np.convolve(y, kernel, mode="same")


def barycenter(x: np.ndarray, y: np.ndarray) -> float:
    """Weighted barycenter for nonnegative weights."""
    y_pos = np.clip(np.asarray(y, dtype=float), 0.0, None)
    mass = np.trapezoid(y_pos, x)
    if mass <= 1e-15:
        return float("nan")
    return float(np.trapezoid(x * y_pos, x) / mass)


def phi(x_value: float, params: LoopParams) -> float:
    """Closed-loop contraction map."""
    return params.x_star + params.lam * (x_value - params.x_star)


def iterate_phi(x0: float, params: LoopParams, n_steps: int) -> np.ndarray:
    xs = [x0]
    x_curr = x0
    for _ in range(n_steps):
        x_curr = phi(x_curr, params)
        xs.append(x_curr)
    return np.asarray(xs, dtype=float)


def readout_profile(x_grid: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Unit-mass Gaussian profile centered at `center`."""
    profile = np.exp(-((x_grid - center) ** 2) / (2.0 * sigma**2))
    profile /= np.trapezoid(profile, x_grid)
    return profile


def exact_decodability_score(x_raw: np.ndarray, y_raw: np.ndarray, y_hammered: np.ndarray) -> float:
    """
    Toy decodability score in [0,1].
    1 means the dominant peak location survived perfectly.
    """
    idx_raw = int(np.argmax(np.abs(y_raw)))
    idx_ham = int(np.argmax(np.abs(y_hammered)))
    if len(x_raw) <= 1:
        return 1.0
    dx = float(np.max(x_raw) - np.min(x_raw))
    if dx <= 0:
        return 1.0
    peak_shift = abs(float(x_raw[idx_raw] - x_raw[idx_ham])) / dx
    return float(max(0.0, 1.0 - peak_shift))


def estimate_noise_energy(y_raw: np.ndarray, y_hammered: np.ndarray) -> float:
    """Toy estimate of removed high-frequency energy."""
    residual = y_raw - y_hammered
    return float(np.mean(residual**2))


def make_synthetic_signal(kind: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n)

    if kind == "Spike + noise":
        y = 0.15 * rng.normal(size=n)
        y += np.exp(-0.5 * ((x - 5.0) / 0.08) ** 2) * 3.0
        y += 0.2 * np.sin(4.0 * x)
    elif kind == "Two peaks + noise":
        y = 0.12 * rng.normal(size=n)
        y += 1.8 * np.exp(-0.5 * ((x - 3.0) / 0.25) ** 2)
        y += 1.2 * np.exp(-0.5 * ((x - 7.3) / 0.45) ** 2)
    elif kind == "Step + impulse":
        y = np.where(x > 4.0, 1.0, 0.0)
        y += 2.5 * np.exp(-0.5 * ((x - 6.8) / 0.05) ** 2)
        y += 0.08 * rng.normal(size=n)
    else:
        y = np.sin(x) + 0.3 * np.sin(8.0 * x) + 0.15 * rng.normal(size=n)

    return x, y


def parse_uploaded_csv(file_obj: io.BytesIO) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_obj)
    if df.shape[1] == 1:
        y = df.iloc[:, 0].astype(float).to_numpy()
        x = np.arange(len(y), dtype=float)
    else:
        x = df.iloc[:, 0].astype(float).to_numpy()
        y = df.iloc[:, 1].astype(float).to_numpy()
    if len(x) < 5:
        raise ValueError("Need at least 5 samples")
    return x, y


def unitary_fft(v: np.ndarray) -> np.ndarray:
    """Unitary discrete Fourier transform."""
    return np.fft.fft(np.asarray(v, dtype=complex), norm="ortho")


def fourier_powers(v: np.ndarray) -> list[np.ndarray]:
    """Return [F^0 v, F^1 v, F^2 v, F^3 v] for the unitary DFT."""
    f0 = np.asarray(v, dtype=complex)
    f1 = unitary_fft(f0)
    f2 = unitary_fft(f1)
    f3 = unitary_fft(f2)
    return [f0, f1, f2, f3]


def fourier_sector_projectors(v: np.ndarray) -> dict[str, np.ndarray]:
    """
    Spectral projectors for the four Fourier sectors:
        P_lambda = (1/4) * sum_{m=0}^3 lambda^{-m} F^m.
    """
    powers = fourier_powers(v)
    sectors: dict[str, np.ndarray] = {}
    for key, lam in FOURIER_EIGENVALUES.items():
        proj = sum((lam ** (-m)) * powers[m] for m in range(4)) / 4.0
        sectors[key] = proj
    return sectors


def energy(v: np.ndarray) -> float:
    """L2 energy of a discrete vector."""
    vv = np.asarray(v, dtype=complex)
    return float(np.vdot(vv, vv).real)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error."""
    denom = np.linalg.norm(np.asarray(b, dtype=complex))
    if denom <= 1e-15:
        return float(np.linalg.norm(np.asarray(a, dtype=complex) - np.asarray(b, dtype=complex)))
    return float(np.linalg.norm(np.asarray(a, dtype=complex) - np.asarray(b, dtype=complex)) / denom)


def main() -> None:
    st.title("AnchorLab")
    st.caption("Raw signal -> Gaussian hammer -> stable readout -> anchor")

    with st.sidebar:
        st.header("Input")
        mode = st.radio("Source", ["Synthetic", "CSV upload"], index=0)

        if mode == "Synthetic":
            kind = st.selectbox(
                "Signal type",
                ["Spike + noise", "Two peaks + noise", "Step + impulse", "Oscillatory"],
            )
            n = st.slider("Samples", 200, 4000, 1200, 50)
            seed = st.number_input("Seed", min_value=0, max_value=999999, value=42)
            x, y = make_synthetic_signal(kind, n, int(seed))
        else:
            uploaded = st.file_uploader("Upload CSV (x,y) or single-column y", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV to continue.")
                st.stop()
            try:
                x, y = parse_uploaded_csv(uploaded)
            except Exception as exc:
                st.error(f"CSV parse failed: {exc}")
                st.stop()

        st.header("Hammer")
        sigma_signal = st.slider("Signal hammer σ", 0.2, 20.0, 2.0, 0.1)

        st.header("Closed loop")
        alpha = float(np.min(x))
        beta = float(np.max(x))
        x_star = st.slider("Anchor x*", float(alpha), float(beta), float((alpha + beta) / 2.0), 0.01)
        lam = st.slider("Contraction λ", 0.01, 0.99, 0.40, 0.01)
        sigma_readout = st.slider("Readout σ", 0.05, 2.0, 0.30, 0.01)
        n_iter = st.slider("Iterations", 1, 30, 10, 1)

        st.header("Fourier sectors")
        sector_source = st.radio("Decompose", ["Hammered signal", "Raw signal"], index=0)
        sector_plot_mode = st.radio("Sector plot", ["Real part", "Magnitude"], index=1)

    params = LoopParams(alpha=alpha, beta=beta, x_star=x_star, lam=lam, sigma=sigma_readout)

    # Core hammer + loop
    y_h = gaussian_hammer(y, sigma_signal)
    raw_anchor = barycenter(x, np.abs(y))
    hammered_anchor = barycenter(x, np.abs(y_h))

    if np.isnan(hammered_anchor):
        st.error("Hammered signal has zero mass after clipping. Try a different input.")
        st.stop()

    iters = iterate_phi(hammered_anchor, params, n_iter)
    x_fixed = params.x_star
    r_star = readout_profile(x, x_fixed, params.sigma)

    q_exact = params.lam
    xg_for_q = np.linspace(alpha, beta, 200)
    q_numeric = float(
        np.max(
            np.abs(np.diff([phi(val, params) for val in xg_for_q])) / np.diff(xg_for_q)
        )
    )
    decodability = exact_decodability_score(x, y, y_h)
    removed_noise = estimate_noise_energy(y, y_h)

    # Fourier sectors
    sector_signal = y_h if sector_source == "Hammered signal" else y
    sector_signal_complex = np.asarray(sector_signal, dtype=complex)
    sectors = fourier_sector_projectors(sector_signal_complex)
    reconstructed = sum(sectors.values())
    reconstruction_error = relative_error(reconstructed, sector_signal_complex)

    sector_energies = {key: energy(val) for key, val in sectors.items()}
    self_dual_error = relative_error(unitary_fft(sectors["1"]), sectors["1"])
    sector_df = pd.DataFrame(
        {
            "sector": list(sector_energies.keys()),
            "eigenvalue": ["1", "-i", "-1", "i"],
            "energy": list(sector_energies.values()),
        }
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw anchor", f"{raw_anchor:.6f}" if not np.isnan(raw_anchor) else "nan")
    col2.metric("Hammered anchor", f"{hammered_anchor:.6f}")
    col3.metric("Exact q", f"{q_exact:.3f}")
    col4.metric("Numeric q", f"{q_numeric:.3f}")

    st.markdown(
        f"""
### Brutal summary

\\[
\\Phi_\\sigma(x)=x_*+\\lambda(x-x_*)={params.lam:.3f}x+{params.x_star * (1.0 - params.lam):.3f},
\\qquad
q={params.lam:.3f}<1,
\\qquad
x_*={params.x_star:.3f}.
\\]

The loop contracts to the anchor, and the fixed-point readout is a nonzero Gaussian profile.

### Fourier-sector decomposition

For the unitary discrete Fourier operator \\(\\mathcal F\\), define

\\[
P_\\lambda = \\frac14 \\sum_{{m=0}}^3 \\lambda^{{-m}}\\mathcal F^m,
\\qquad
\\lambda\\in\\{{1,-i,-1,i\\}}.
\\]

Then the selected signal decomposes as

\\[
f=P_1f+P_{{-i}}f+P_{{-1}}f+P_if,
\\]

and \\(P_1f\\) is the self-dual core.
"""
    )

    # Plot 1: Raw vs hammered signal
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(x, y, label="raw", linewidth=1.0)
    ax1.plot(x, y_h, label="hammered", linewidth=2.0)
    ax1.axvline(raw_anchor, linestyle="--", linewidth=1.2, label="raw anchor")
    ax1.axvline(hammered_anchor, linestyle=":", linewidth=1.6, label="hammered anchor")
    ax1.set_title("Raw signal vs Gaussian hammer")
    ax1.set_xlabel("x")
    ax1.set_ylabel("signal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Phi and identity
    xg = np.linspace(alpha, beta, 200)
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(xg, xg, label="identity")
    ax2.plot(xg, [phi(v, params) for v in xg], label=r"$\Phi_\sigma(x)$")
    ax2.scatter([x_fixed], [x_fixed], s=70, label="fixed point")
    ax2.set_title("Closed-loop map")
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$\Phi_\sigma(x)$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Iterations
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    ax3.plot(range(len(iters)), iters, marker="o")
    ax3.axhline(x_fixed, linestyle="--", linewidth=1.3, label="x*")
    ax3.set_title("Iterates collapse to the anchor")
    ax3.set_xlabel("iteration n")
    ax3.set_ylabel(r"$x_n$")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Fixed-point readout
    fig4, ax4 = plt.subplots(figsize=(9, 4.5))
    ax4.plot(x, r_star, label=r"$r_*=G_\sigma(\cdot-x_*)$")
    ax4.axvline(x_fixed, linestyle="--", linewidth=1.2, label="x*")
    ax4.set_title("Fixed-point readout is nonzero")
    ax4.set_xlabel("x")
    ax4.set_ylabel("readout")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    row1, row2 = st.columns(2)
    row1.pyplot(fig1)
    row2.pyplot(fig2)

    row3, row4 = st.columns(2)
    row3.pyplot(fig3)
    row4.pyplot(fig4)

    # Fourier sector plots
    st.subheader("Fourier sectors")
    fig5, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    sector_keys = ["1", "-i", "-1", "i"]
    for ax, key in zip(axes.flat, sector_keys):
        component = sectors[key]
        if sector_plot_mode == "Real part":
            ax.plot(x, np.real(component), label=f"Re P_{key} f")
        else:
            ax.plot(x, np.abs(component), label=f"|P_{key} f|")
        ax.set_title(f"Sector {key}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("component")
    plt.tight_layout()
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(7, 4.5))
    ax6.bar(sector_df["sector"], sector_df["energy"])
    ax6.set_title("Energy by Fourier sector")
    ax6.set_xlabel("sector")
    ax6.set_ylabel("energy")
    ax6.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig6)

    # Tables and diagnostics
    st.subheader("Diagnostics")
    diag = pd.DataFrame(
        {
            "quantity": [
                "removed noise energy",
                "decodability score",
                "fixed-point readout L1",
                "fixed-point readout peak",
                "distance |x_n - x*| after last step",
                "Fourier reconstruction error",
                "self-dual sector error ||F(P1 f)-P1 f|| / ||P1 f||",
            ],
            "value": [
                removed_noise,
                decodability,
                float(np.trapezoid(np.abs(r_star), x)),
                float(np.max(r_star)),
                float(abs(iters[-1] - x_fixed)),
                reconstruction_error,
                self_dual_error,
            ],
        }
    )
    st.dataframe(diag, use_container_width=True)

    iter_df = pd.DataFrame(
        {
            "n": np.arange(len(iters)),
            "x_n": iters,
            "|x_n - x*|": np.abs(iters - x_fixed),
        }
    )
    st.subheader("Iteration table")
    st.dataframe(iter_df, use_container_width=True)

    st.subheader("Sector energies")
    st.dataframe(sector_df, use_container_width=True)

    # Verdict
    st.subheader("Verdict")
    if q_exact < 1.0 and np.max(r_star) > 0 and reconstruction_error < 1e-10:
        st.success(
            "The hammer smooths. The barycenter returns. The loop contracts. Collapse is vetoed. Fourier sectors are resolved."
        )
    else:
        st.warning(
            "This parameter regime certifies only part of the pipeline. Check contraction, readout, or Fourier-sector diagnostics."
        )


if __name__ == "__main__":
    main()
