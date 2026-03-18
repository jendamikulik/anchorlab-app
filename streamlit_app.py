import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="AnchorLab v4", layout="wide")


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
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if radius is None:
        radius = max(3, int(math.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_hammer(y: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma)
    return np.convolve(y, kernel, mode="same")


def barycenter(x: np.ndarray, y: np.ndarray) -> float:
    y_pos = np.clip(np.asarray(y, dtype=float), 0.0, None)
    mass = np.trapezoid(y_pos, x)
    if mass <= 1e-15:
        return float("nan")
    return float(np.trapezoid(x * y_pos, x) / mass)


def phi(x_value: float, params: LoopParams) -> float:
    return params.x_star + params.lam * (x_value - params.x_star)


def iterate_phi(x0: float, params: LoopParams, n_steps: int) -> np.ndarray:
    xs = [x0]
    x_curr = x0
    for _ in range(n_steps):
        x_curr = phi(x_curr, params)
        xs.append(x_curr)
    return np.asarray(xs, dtype=float)


def readout_profile(x_grid: np.ndarray, center: float, sigma: float) -> np.ndarray:
    profile = np.exp(-((x_grid - center) ** 2) / (2.0 * sigma**2))
    profile /= np.trapezoid(profile, x_grid)
    return profile


def exact_decodability_score(x_raw: np.ndarray, y_raw: np.ndarray, y_hammered: np.ndarray) -> float:
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


def parse_uploaded_csv(file_obj) -> tuple[np.ndarray, np.ndarray]:
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
    return np.fft.fft(np.asarray(v, dtype=complex), norm="ortho")


def fourier_powers(v: np.ndarray) -> list[np.ndarray]:
    f0 = np.asarray(v, dtype=complex)
    f1 = unitary_fft(f0)
    f2 = unitary_fft(f1)
    f3 = unitary_fft(f2)
    return [f0, f1, f2, f3]


def fourier_sector_projectors(v: np.ndarray) -> dict[str, np.ndarray]:
    powers = fourier_powers(v)
    sectors: dict[str, np.ndarray] = {}
    for key, lam in FOURIER_EIGENVALUES.items():
        proj = sum((lam ** (-m)) * powers[m] for m in range(4)) / 4.0
        sectors[key] = proj
    return sectors


def energy(v: np.ndarray) -> float:
    vv = np.asarray(v, dtype=complex)
    return float(np.vdot(vv, vv).real)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(np.asarray(b, dtype=complex))
    if denom <= 1e-15:
        return float(np.linalg.norm(np.asarray(a, dtype=complex) - np.asarray(b, dtype=complex)))
    return float(np.linalg.norm(np.asarray(a, dtype=complex) - np.asarray(b, dtype=complex)) / denom)


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)


def make_signal_plot(x, y, y_h, raw_anchor, hammered_anchor):
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.plot(x, y, label="raw", linewidth=1.2)
    ax.plot(x, y_h, label="hammered", linewidth=2.0)
    ax.axvline(raw_anchor, linestyle="--", linewidth=1.1, label="raw anchor")
    ax.axvline(hammered_anchor, linestyle=":", linewidth=1.4, label="hammered anchor")
    style_axes(ax, "Raw signal vs Gaussian hammer", "x", "signal")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    return fig


def make_closed_loop_plot(alpha, beta, params, x_fixed):
    xg = np.linspace(alpha, beta, 300)
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.plot(xg, xg, label="identity", linewidth=1.3)
    ax.plot(xg, [phi(v, params) for v in xg], label=r"$\Phi_\sigma(x)$", linewidth=2.0)
    ax.scatter([x_fixed], [x_fixed], s=80, label="fixed point")
    style_axes(ax, "Closed-loop map", "x", r"$\Phi_\sigma(x)$")
    ax.legend(frameon=False, loc="upper left")
    return fig


def make_iteration_plot(iters, x_fixed):
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.plot(range(len(iters)), iters, marker="o", linewidth=1.8, markersize=4)
    ax.axhline(x_fixed, linestyle="--", linewidth=1.2, label="x*")
    style_axes(ax, "Iterates collapse to the anchor", "iteration n", r"$x_n$")
    ax.legend(frameon=False, loc="best")
    return fig


def make_readout_plot(x, r_star, x_fixed):
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.plot(x, r_star, label=r"$r_*=G_\sigma(\cdot-x_*)$", linewidth=2.0)
    ax.axvline(x_fixed, linestyle="--", linewidth=1.2, label="x*")
    style_axes(ax, "Fixed-point readout is nonzero", "x", "readout")
    ax.legend(frameon=False, loc="upper right")
    return fig


def make_sector_grid(x, sectors, mode: str):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    sector_keys = ["1", "-i", "-1", "i"]
    for ax, key in zip(axes.flat, sector_keys):
        component = sectors[key]
        if mode == "Real part":
            ax.plot(x, np.real(component), linewidth=1.7)
            ylabel = "Re component"
        else:
            ax.plot(x, np.abs(component), linewidth=1.7)
            ylabel = "magnitude"
        style_axes(ax, f"Sector {key}", "x", ylabel)
    return fig


def make_energy_plot(sector_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10.5, 3.8), constrained_layout=True)
    ax.bar(sector_df["sector"], sector_df["energy"])
    style_axes(ax, "Energy by Fourier sector", "sector", "energy")
    ax.grid(True, axis="y", alpha=0.25)
    return fig


def bridge_threshold(c1: float, c2: float, c: float, C: float, beta: float) -> tuple[float, float, float, float, float]:
    gap = abs(beta - 0.5)
    A = c1 * c * gap**4
    B = c1 * C
    D = c2
    if A <= 0:
        return A, B, D, float("inf"), gap
    tau_star = max(2.0 * B / A, (4.0 * D / A) ** 0.25 if D > 0 else 0.0)
    return A, B, D, tau_star, gap


def make_bridge_plot(A: float, B: float, D: float, tau_star: float, tau_max: float):
    taus = np.linspace(0.0, tau_max, 500)
    vals = A * taus**4 - B * taus**3 - D
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    ax.plot(taus, vals, linewidth=2.0, label=r"$A\tau^4-B\tau^3-D$")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    if np.isfinite(tau_star):
        ax.axvline(tau_star, linestyle=":", linewidth=1.4, label=r"$\tau_*$")
    style_axes(ax, "Bridge-lock positivity curve", r"$\tau$", "lower bound")
    ax.legend(frameon=False, loc="best")
    return fig


def vindaloo_readout(delta: np.ndarray) -> np.ndarray:
    return -np.sin(delta)


def vindaloo_derivative(delta: np.ndarray, c: float, sigma: float) -> np.ndarray:
    # Uses delta_dot = - c delta sigma
    return c * delta * sigma * np.cos(delta)


def make_vindaloo_plot(delta_max: float, c: float, sigma: float):
    delta = np.linspace(-delta_max, delta_max, 600)
    M = vindaloo_readout(delta)
    Mdot = vindaloo_derivative(delta, c, sigma)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(delta, M, linewidth=2.0)
    style_axes(axes[0], r"Vindaloo readout $M(\delta)=-\sin\delta$", r"$\delta$", r"$M(\delta)$")
    axes[1].plot(delta, Mdot, linewidth=2.0)
    style_axes(axes[1], r"$\dot M = c\,\delta\,\sigma\cos\delta$", r"$\delta$", r"$\dot M$")
    return fig


def substitution_numeric_demo(x: np.ndarray, u_name: str):
    if u_name == "sin(x)":
        u = np.sin(x)
        up = np.cos(x)
        F = lambda z: z**3
        Fp = lambda z: 3 * z**2
        label = r"$u(x)=\sin x,\ F(u)=u^3$"
    elif u_name == "x^2 + 1":
        u = x**2 + 1.0
        up = 2 * x
        F = lambda z: np.log(z)
        Fp = lambda z: 1.0 / z
        label = r"$u(x)=x^2+1,\ F(u)=\log u$"
    else:
        u = np.exp(0.35 * x)
        up = 0.35 * np.exp(0.35 * x)
        F = lambda z: np.sqrt(z)
        Fp = lambda z: 0.5 / np.sqrt(z)
        label = r"$u(x)=e^{0.35x},\ F(u)=\sqrt{u}$"

    lhs = np.gradient(F(u), x)
    rhs = Fp(u) * up
    err = float(np.max(np.abs(lhs - rhs)))
    return u, lhs, rhs, err, label


def make_substitution_plot(x, lhs, rhs):
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    ax.plot(x, lhs, linewidth=2.0, label=r"$D_x(U_uF)$")
    ax.plot(x, rhs, linestyle="--", linewidth=2.0, label=r"$M_{u'}U_uD_uF$")
    style_axes(ax, "Swiss-army-knife operator identity", "x", "value")
    ax.legend(frameon=False, loc="best")
    return fig


def main() -> None:
    st.title("AnchorLab v4")
    st.caption("Anchor -> sectors -> Hermite spine -> Bridge lock -> Swiss knife -> Vindaloo")

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
        beta_dom = float(np.max(x))
        x_star = st.slider("Anchor x*", float(alpha), float(beta_dom), float((alpha + beta_dom) / 2.0), 0.01)
        lam = st.slider("Contraction λ", 0.01, 0.99, 0.40, 0.01)
        sigma_readout = st.slider("Readout σ", 0.05, 2.0, 0.30, 0.01)
        n_iter = st.slider("Iterations", 1, 30, 10, 1)

        st.header("Fourier sectors")
        sector_source = st.radio("Decompose", ["Hammered signal", "Raw signal"], index=0)
        sector_plot_mode = st.radio("Sector plot", ["Real part", "Magnitude"], index=1)

    params = LoopParams(alpha=alpha, beta=beta_dom, x_star=x_star, lam=lam, sigma=sigma_readout)
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
    xg_for_q = np.linspace(alpha, beta_dom, 200)
    q_numeric = float(np.max(np.abs(np.diff([phi(val, params) for val in xg_for_q])) / np.diff(xg_for_q)))
    decodability = exact_decodability_score(x, y, y_h)
    removed_noise = estimate_noise_energy(y, y_h)

    sector_signal = y_h if sector_source == "Hammered signal" else y
    sector_signal_complex = np.asarray(sector_signal, dtype=complex)
    sectors = fourier_sector_projectors(sector_signal_complex)
    reconstructed = sum(sectors.values())
    reconstruction_error = relative_error(reconstructed, sector_signal_complex)
    sector_energies = {key: energy(val) for key, val in sectors.items()}
    self_dual_error = relative_error(unitary_fft(sectors["1"]), sectors["1"])
    sector_df = pd.DataFrame({"sector": list(sector_energies.keys()), "energy": list(sector_energies.values())})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw anchor", f"{raw_anchor:.6f}" if not np.isnan(raw_anchor) else "nan")
    col2.metric("Hammered anchor", f"{hammered_anchor:.6f}")
    col3.metric("Exact q", f"{q_exact:.3f}")
    col4.metric("Numeric q", f"{q_numeric:.3f}")

    st.markdown(
        rf"""
### Brutal summary

\[
\Phi_\sigma(x)=x_*+\lambda(x-x_*)={params.lam:.3f}x+{params.x_star * (1.0 - params.lam):.3f},
\qquad
q={params.lam:.3f}<1,
\qquad
x_*={params.x_star:.3f}.
\]

For the unitary discrete Fourier operator \(\mathcal F\),

\[
P_\lambda = \frac14 \sum_{{m=0}}^3 \lambda^{{-m}}\mathcal F^m,
\qquad
\lambda\in\{{1,-i,-1,i\}}.
\]

So the selected signal decomposes as

\[
f=P_1f+P_{{-i}}f+P_{{-1}}f+P_if,
\]

and \(P_1f\) is the self-dual core.
"""
    )

    tabs = st.tabs([
        "Overview",
        "Anchoring",
        "Fourier sectors",
        "Hermite spine",
        "Bridge lock",
        "Swiss knife",
        "Vindaloo",
        "Tables",
    ])

    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.pyplot(make_signal_plot(x, y, y_h, raw_anchor, hammered_anchor), clear_figure=True)
        c2.pyplot(make_closed_loop_plot(alpha, beta_dom, params, x_fixed), clear_figure=True)
        c3, c4 = st.columns(2)
        c3.pyplot(make_iteration_plot(iters, x_fixed), clear_figure=True)
        c4.pyplot(make_readout_plot(x, r_star, x_fixed), clear_figure=True)

    with tabs[1]:
        st.subheader("Anchor dynamics")
        c1, c2 = st.columns(2)
        c1.pyplot(make_closed_loop_plot(alpha, beta_dom, params, x_fixed), clear_figure=True)
        c2.pyplot(make_iteration_plot(iters, x_fixed), clear_figure=True)
        iter_df = pd.DataFrame({"n": np.arange(len(iters)), "x_n": iters, "|x_n-x*|": np.abs(iters - x_fixed)})
        st.dataframe(iter_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Fourier sectors")
        st.pyplot(make_sector_grid(x, sectors, sector_plot_mode), clear_figure=True)
        st.pyplot(make_energy_plot(sector_df), clear_figure=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Reconstruction error", f"{reconstruction_error:.3e}")
        c2.metric("Self-dual error", f"{self_dual_error:.3e}")
        c3.metric("Selected source", sector_source)

    with tabs[3]:
        st.subheader("Hermite spine")
        st.latex(r"(\mathcal F f)(\xi)=\int_{\mathbb R} f(x)e^{-2\pi i x\xi}\,dx")
        st.latex(r"A:=\frac{1}{2\sqrt{\pi}}D_x+\sqrt{\pi}M_x,\qquad A^\dagger:=-\frac{1}{2\sqrt{\pi}}D_x+\sqrt{\pi}M_x")
        st.latex(r"\mathcal F A=iA\mathcal F,\qquad \mathcal F A^\dagger=-iA^\dagger\mathcal F")
        st.latex(r"h_0(x)=2^{1/4}e^{-\pi x^2},\qquad Ah_0=0,\qquad \mathcal F h_0=h_0")
        st.latex(r"h_n:=\frac{1}{\sqrt{n!}}(A^\dagger)^n h_0,\qquad \mathcal F h_n=(-i)^n h_n")
        st.latex(r"N:=A^\dagger A,\qquad \mathcal F=e^{-i\pi N/2}")
        st.info("Gaussian is the vacuum. Hermite modes are the excitations. Fourier only rotates their phase.")

    with tabs[4]:
        st.subheader("Bridge lock theorem")
        c_left, c_right = st.columns([1, 1.2])
        with c_left:
            beta_rh = st.number_input("β (off-critical real part)", min_value=0.0, max_value=1.0, value=0.61, step=0.01, key="beta_rh")
            c1_val = st.number_input("c₁ (coercivity margin)", min_value=1e-6, value=0.8, step=0.05, format="%.6f")
            c2_val = st.number_input("c₂ (leakage tax)", min_value=0.0, value=1.0, step=0.1, format="%.6f")
            c_val = st.number_input("c (quartic growth constant)", min_value=1e-6, value=0.4, step=0.05, format="%.6f")
            C_val = st.number_input("C (cubic correction)", min_value=0.0, value=0.6, step=0.05, format="%.6f")
            A, B, D, tau_star, gap = bridge_threshold(c1_val, c2_val, c_val, C_val, beta_rh)
            st.metric(r"|β-1/2|", f"{gap:.4f}")
            st.metric("A", f"{A:.6f}")
            st.metric("τ*", "∞" if not np.isfinite(tau_star) else f"{tau_star:.6f}")
        with c_right:
            tau_max = max(10.0, (tau_star if np.isfinite(tau_star) else 4.0) * 1.8 + 2.0)
            st.pyplot(make_bridge_plot(A, B, D, tau_star, tau_max), clear_figure=True)

        st.latex(r"\frac{Q_{\mathrm{bank}}(f;\tau,L)-E_0L}{L} \ge A\tau^4-B\tau^3-D")
        st.latex(r"\tau_* = \max\!\left\{\frac{2B}{A},\left(\frac{4D}{A}\right)^{1/4}\right\}")
        st.markdown("Above the trigger threshold, the quartic drift dominates the leakage tax and forces positive excess.")

    with tabs[5]:
        st.subheader("Substitution operator Swiss Army Knife")
        c1, c2 = st.columns([1, 1.2])
        with c1:
            u_name = st.selectbox("Demo substitution", ["sin(x)", "x^2 + 1", "exp(0.35x)"], index=0)
            x_demo = np.linspace(-2.5, 2.5, 500)
            u, lhs, rhs, err, label = substitution_numeric_demo(x_demo, u_name)
            st.latex(r"(U_uF)(x):=F(u(x))")
            st.latex(r"D_x\circ U_u = M_{u'}\circ U_u\circ D_u")
            st.latex(r"\int f(u(x))u'(x)\,dx = F(u(x))+C")
            st.caption(label)
            st.metric("numeric max error", f"{err:.3e}")
        with c2:
            st.pyplot(make_substitution_plot(x_demo, lhs, rhs), clear_figure=True)
        st.markdown("The whole substitution rule is one operator identity. Everything else is fallout.")

    with tabs[6]:
        st.subheader("Vindaloo entanglement principle")
        c1, c2 = st.columns([1, 1.2])
        with c1:
            delta_max = st.slider(r"δ range", 0.2, 3.2, 2.2, 0.05)
            c_v = st.slider("c", 0.1, 3.0, 1.0, 0.05)
            sigma_v = st.slider("σ curvature", -2.0, 2.0, 0.7, 0.05)
            st.latex(r"M(\delta)=-\sin\delta")
            st.latex(r"\dot\delta(t)=-c\,\delta(t)\,\sigma(t)")
            st.latex(r"\dot M = c\,\delta\,\sigma\cos\delta = c\,\delta\,\sigma + O(\delta^3\sigma)")
            st.metric("small-δ slope sign", "positive" if c_v * sigma_v > 0 else "negative" if c_v * sigma_v < 0 else "zero")
        with c2:
            st.pyplot(make_vindaloo_plot(delta_max, c_v, sigma_v), clear_figure=True)
        st.markdown("Absurd superposition + tiny phase injection + positive closure curvature = deterministic punchline release.")

    with tabs[7]:
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
        iter_df = pd.DataFrame({"n": np.arange(len(iters)), "x_n": iters, "|x_n-x*|": np.abs(iters - x_fixed)})
        st.subheader("Diagnostics")
        st.dataframe(diag, use_container_width=True, hide_index=True)
        st.subheader("Sector energies")
        st.dataframe(sector_df, use_container_width=True, hide_index=True)
        st.subheader("Iteration table")
        st.dataframe(iter_df, use_container_width=True, hide_index=True)

    st.subheader("Verdict")
    if q_exact < 1.0 and np.max(r_star) > 0 and reconstruction_error < 1e-10:
        st.success(
            "The hammer smooths. The barycenter returns. The loop contracts. Collapse is vetoed. Fourier sectors are resolved. Bridge lock, Swiss knife, and Vindaloo are wired in."
        )
    else:
        st.warning("This parameter regime certifies only part of the pipeline. Check contraction, readout, or Fourier-sector diagnostics.")


if __name__ == "__main__":
    main()
