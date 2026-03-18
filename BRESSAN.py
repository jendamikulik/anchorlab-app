
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class KernelData:
    x: np.ndarray
    y: np.ndarray
    dx: float
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    K11: np.ndarray
    K12: np.ndarray
    K21: np.ndarray
    K22: np.ndarray
    K11c: np.ndarray
    K12c: np.ndarray
    K21c: np.ndarray
    K22c: np.ndarray


def log_weights(rs: np.ndarray) -> np.ndarray:
    logs = np.log(rs)
    w = np.zeros_like(rs)
    w[0] = 0.5 * (logs[1] - logs[0])
    w[-1] = 0.5 * (logs[-1] - logs[-2])
    w[1:-1] = 0.5 * (logs[2:] - logs[:-2])
    return w


def gaussian_eta(X: np.ndarray, Y: np.ndarray, r: float) -> np.ndarray:
    rr2 = X * X + Y * Y
    return (1.0 / (2.0 * np.pi * r * r)) * np.exp(-0.5 * rr2 / (r * r))


def build_kernels(
    N: int = 301,
    L: float = 10.0,
    eps: float = 0.12,
    R: float = 2.0,
    nr: int = 120,
) -> KernelData:
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="xy")

    rs = np.geomspace(eps, R, nr)
    w = log_weights(rs)

    dPhi_x = np.zeros_like(X)
    dPhi_y = np.zeros_like(X)

    for wi, r in zip(w, rs):
        eta = gaussian_eta(X, Y, r)
        dPhi_x += wi * (-(X / (r * r)) * eta)
        dPhi_y += wi * (-(Y / (r * r)) * eta)

    Z = X * dPhi_x + Y * dPhi_y

    K11 = X * dPhi_x
    K12 = X * dPhi_y
    K21 = Y * dPhi_x
    K22 = Y * dPhi_y

    K11c = K11 - 0.5 * Z
    K22c = K22 - 0.5 * Z
    K12c = K12
    K21c = K21

    return KernelData(
        x=x, y=y, dx=dx, X=X, Y=Y, Z=Z,
        K11=K11, K12=K12, K21=K21, K22=K22,
        K11c=K11c, K12c=K12c, K21c=K21c, K22c=K22c
    )


def integrate(field: np.ndarray, dx: float) -> float:
    return float(field.sum() * dx * dx)


def contract_full(kd: KernelData, B: np.ndarray) -> np.ndarray:
    # H(z) = sum_{j,k} B_{k j} K_{j k}(z)
    return (
        B[0, 0] * kd.K11 + B[0, 1] * kd.K21 +
        B[1, 0] * kd.K12 + B[1, 1] * kd.K22
    )


def contract_traceless(kd: KernelData, B: np.ndarray) -> np.ndarray:
    return (
        B[0, 0] * kd.K11c + B[0, 1] * kd.K21c +
        B[1, 0] * kd.K12c + B[1, 1] * kd.K22c
    )


def contract_contact(kd: KernelData, B: np.ndarray) -> np.ndarray:
    return 0.5 * np.trace(B) * kd.Z


def fft_convolve_same(kernel: np.ndarray, f: np.ndarray, dx: float) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(kernel) * np.fft.fft2(f)).real * (dx * dx)


def run_demo() -> tuple[str, list[tuple[float, float]], KernelData]:
    eps = 0.12
    R = 2.0
    kd = build_kernels(eps=eps, R=R)

    lines: list[str] = []
    lines.append("Bressan core numerics")
    lines.append("=====================")
    lines.append("")
    lines.append(f"grid: N={kd.X.shape[0]}, box=[-10,10]^2, dx={kd.dx:.6f}")
    lines.append(f"kernel parameters: eps={eps}, R={R}")
    lines.append("")

    # Structural identities
    trace_full = kd.K11 + kd.K22
    trace_traceless = kd.K11c + kd.K22c

    mass_Z = integrate(kd.Z, kd.dx)
    target_mass = -2.0 * np.log(R / eps)
    lines.append("Structural checks")
    lines.append("-----------------")
    lines.append(f"max |trace(K)-Z|               = {np.max(np.abs(trace_full - kd.Z)):.3e}")
    lines.append(f"max |trace(K^o)|               = {np.max(np.abs(trace_traceless)):.3e}")
    lines.append(f"int Z                          = {mass_Z:.12f}")
    lines.append(f"-2 log(R/eps)                  = {target_mass:.12f}")
    lines.append(f"|int Z + 2 log(R/eps)|         = {abs(mass_Z - target_mass):.3e}")
    lines.append("")

    # Matrix contractions
    matrices = {
        "incompressible_hyperbolic": np.array([[1.0, 0.0], [0.0, -1.0]]),
        "incompressible_shear": np.array([[0.0, 1.0], [0.0, 0.0]]),
        "compressible_identity": np.array([[1.0, 0.0], [0.0, 1.0]]),
    }

    lines.append("Contraction checks")
    lines.append("------------------")
    for name, B in matrices.items():
        H_full = contract_full(kd, B)
        H_tr = contract_traceless(kd, B)
        H_ct = contract_contact(kd, B)

        err_split = np.max(np.abs(H_full - (H_tr + H_ct)))
        contact_norm = np.linalg.norm(H_ct)
        full_minus_tr = np.linalg.norm(H_full - H_tr)
        mass_full = integrate(H_full, kd.dx)
        mass_ct = integrate(H_ct, kd.dx)

        lines.append(f"{name}:")
        lines.append(f"  trace(B)                     = {np.trace(B):.6f}")
        lines.append(f"  max |H_full-(H_tr+H_ct)|     = {err_split:.3e}")
        lines.append(f"  ||H_contact||_2              = {contact_norm:.6e}")
        lines.append(f"  ||H_full-H_tr||_2            = {full_minus_tr:.6e}")
        lines.append(f"  int H_full                   = {mass_full:.12f}")
        lines.append(f"  int H_contact                = {mass_ct:.12f}")
        lines.append("")

    # Convolution test with a sample scalar field
    sigma = 0.9
    f = np.exp(-(kd.X * kd.X + kd.Y * kd.Y) / (2.0 * sigma * sigma))
    lines.append("Convolution check with a Gaussian test field")
    lines.append("--------------------------------------------")
    for name in ("incompressible_hyperbolic", "compressible_identity"):
        B = matrices[name]
        H_full = contract_full(kd, B)
        H_tr = contract_traceless(kd, B)
        H_ct = contract_contact(kd, B)

        Cf = fft_convolve_same(H_full, f, kd.dx)
        Ct = fft_convolve_same(H_tr, f, kd.dx)
        Cc = fft_convolve_same(H_ct, f, kd.dx)

        lines.append(f"{name}:")
        lines.append(f"  ||Cf-(Ct+Cc)||_2             = {np.linalg.norm(Cf - (Ct + Cc)):.3e}")
        lines.append(f"  ||Cc||_2                     = {np.linalg.norm(Cc):.6e}")
        lines.append(f"  ||Cf-Ct||_2                  = {np.linalg.norm(Cf - Ct):.6e}")
        lines.append("")

    # Mass law across eps
    eps_values = np.geomspace(0.08, 0.5, 9)
    mass_data: list[tuple[float, float]] = []
    for e in eps_values:
        kde = build_kernels(eps=float(e), R=R, N=241, L=10.0, nr=120)
        mass_data.append((float(np.log(R / e)), float(-0.5 * integrate(kde.Z, kde.dx))))

    return "\n".join(lines), mass_data, kd


def save_plot(mass_data: list[tuple[float, float]], out_path: str | Path) -> None:
    x = np.array([a for a, _ in mass_data])
    y = np.array([b for _, b in mass_data])

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(x, y, marker="o")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(r" \log(R/\varepsilon) ")
    plt.ylabel(r" -\frac12 \int Z_{\varepsilon,R} ")
    plt.title("Mass law for the contact sector")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    report, mass_data, _ = run_demo()
    out_dir = Path("d:/hit/PythonProject")
    (out_dir / "bressan_core_report.txt").write_text(report, encoding="utf-8")
    save_plot(mass_data, "test.png")
    print(report)
    print("")



if __name__ == "__main__":
    main()
