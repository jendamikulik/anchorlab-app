#!/usr/bin/env python3
"""
BRESSAN_SIM_v1.py

Dynamic numerical simulation of the Bressan structural theorem on a 2D box.

What it does
------------
1. Builds the multiscale kernel
       K_{jk}(z) = z_j \partial_k \Phi_{eps,R}(z)
   together with the exact split
       K = K^o + (delta_{jk}/d) Z,
       Z = z · grad Phi.
2. Verifies numerically:
       trace(K) = Z,
       trace(K^o) = 0,
       int Z ~= -d log(R/eps).
3. Evolves a transported scalar rho(t,x) under linear flows x' = Bx:
       rho(t,x) = rho0(exp(-tB) x)
   for
       - incompressible hyperbolic
       - incompressible rotation
       - compressible identity
       - weakly compressible diagonal
4. Computes the dynamic forcing channels
       H_full     = - sum_{jk} B_{jk} (K_{jk} * rho)
       H_traceless= - sum_{jk} B_{jk} (K^o_{jk} * rho)
       H_contact  = -(tr(B)/d) (Z * rho)
   which satisfy
       H_full = H_traceless + H_contact.
5. Tracks in time:
       ||H_full||_2, ||H_traceless||_2, ||H_contact||_2,
       int H_full, int H_contact.
6. Produces plots and a machine-readable report.

This is a structural theorem simulator, not a full numerical proof of the open
Bressan-type lower bound.

Usage
-----
    python BRESSAN_SIM_v1.py
    python BRESSAN_SIM_v1.py --out ./bressan_v1_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Basic utilities
# ============================================================

def gaussian_base(zx: np.ndarray, zy: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized 2D Gaussian eta(z)."""
    r2 = zx**2 + zy**2
    eta = np.exp(-0.5 * r2 / sigma**2) / (2.0 * np.pi * sigma**2)
    return eta


def eta_r(zx: np.ndarray, zy: np.ndarray, sigma: float, r: float) -> np.ndarray:
    """
    eta_r(z) = r^{-d} eta(z/r), here d=2.
    For Gaussian eta, this remains Gaussian with width sigma*r.
    """
    r2 = zx**2 + zy**2
    sr = sigma * r
    return np.exp(-0.5 * r2 / sr**2) / (2.0 * np.pi * sr**2)


def grad_eta_r(zx: np.ndarray, zy: np.ndarray, sigma: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient of eta_r for the Gaussian base.
    """
    er = eta_r(zx, zy, sigma, r)
    sr2 = (sigma * r) ** 2
    return -(zx / sr2) * er, -(zy / sr2) * er


def build_multiscale_kernels(
    x: np.ndarray,
    sigma: float,
    eps: float,
    R: float,
    n_r: int = 80,
) -> dict[str, np.ndarray]:
    """
    Build Phi, grad Phi, K, K^o, Z on a 2D Cartesian grid centered at zero.
    Uses logarithmic quadrature in r for
        Phi(z) = int_eps^R eta_r(z) dr / r.
    """
    X, Y = np.meshgrid(x, x, indexing="xy")
    d = 2

    # logarithmic quadrature in r
    rs = np.geomspace(eps, R, n_r)
    logs = np.log(rs)

    phi = np.zeros_like(X)
    gx = np.zeros_like(X)
    gy = np.zeros_like(X)

    # trapz in log r, using dr/r = d(log r)
    values_phi = []
    values_gx = []
    values_gy = []
    for r in rs:
        values_phi.append(eta_r(X, Y, sigma, r))
        ex, ey = grad_eta_r(X, Y, sigma, r)
        values_gx.append(ex)
        values_gy.append(ey)

    values_phi = np.stack(values_phi, axis=0)
    values_gx = np.stack(values_gx, axis=0)
    values_gy = np.stack(values_gy, axis=0)

    phi = np.trapezoid(values_phi, logs, axis=0)
    gx = np.trapezoid(values_gx, logs, axis=0)
    gy = np.trapezoid(values_gy, logs, axis=0)

    # Full kernel K_jk = z_j * d_k Phi
    K11 = X * gx
    K12 = X * gy
    K21 = Y * gx
    K22 = Y * gy

    Z = X * gx + Y * gy

    # Traceless split
    K11o = K11 - 0.5 * Z
    K22o = K22 - 0.5 * Z
    K12o = K12
    K21o = K21

    return {
        "X": X,
        "Y": Y,
        "Phi": phi,
        "gradx": gx,
        "grady": gy,
        "Z": Z,
        "K11": K11,
        "K12": K12,
        "K21": K21,
        "K22": K22,
        "K11o": K11o,
        "K12o": K12o,
        "K21o": K21o,
        "K22o": K22o,
    }


def l2_norm(f: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum(np.abs(f) ** 2) * dx * dx))


def integral2d(f: np.ndarray, dx: float) -> float:
    return float(np.sum(f) * dx * dx)


def fft_conv_same(kernel: np.ndarray, field: np.ndarray) -> np.ndarray:
    """
    Convolution via FFT with 'same' alignment.
    Arrays must have the same shape and be centered consistently.
    """
    fk = np.fft.fft2(np.fft.ifftshift(kernel))
    ff = np.fft.fft2(field)
    out = np.fft.ifft2(fk * ff).real
    return out


def gaussian_bump(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, sigma: float) -> np.ndarray:
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    return np.exp(-0.5 * r2 / sigma**2)


def rho_transport_linear(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    B: np.ndarray,
    cx0: float,
    cy0: float,
    sigma: float,
) -> np.ndarray:
    """
    Exact solution of
        partial_t rho + (Bx)·grad rho = 0
    with rho(0,x)=rho0(x), using backward characteristic x0 = exp(-tB)x.
    """
    evals, evecs = np.linalg.eig(B)
    evecs_inv = np.linalg.inv(evecs)
    exp_minus_tB = (evecs @ np.diag(np.exp(-t * evals)) @ evecs_inv).real

    X0 = exp_minus_tB[0, 0] * X + exp_minus_tB[0, 1] * Y
    Y0 = exp_minus_tB[1, 0] * X + exp_minus_tB[1, 1] * Y
    return gaussian_bump(X0, Y0, cx0, cy0, sigma)


def dynamic_channels(
    kernels: dict[str, np.ndarray],
    rho: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For constant gradient B_jk = partial_j b_k, the time integral in s disappears.
    Then
        H_full      = - sum B_jk (K_jk * rho)
        H_traceless = - sum B_jk (K^o_jk * rho)
        H_contact   = -(tr B / d) (Z * rho), d=2
    """
    K11rho = fft_conv_same(kernels["K11"], rho)
    K12rho = fft_conv_same(kernels["K12"], rho)
    K21rho = fft_conv_same(kernels["K21"], rho)
    K22rho = fft_conv_same(kernels["K22"], rho)

    K11orho = fft_conv_same(kernels["K11o"], rho)
    K12orho = fft_conv_same(kernels["K12o"], rho)
    K21orho = fft_conv_same(kernels["K21o"], rho)
    K22orho = fft_conv_same(kernels["K22o"], rho)

    Zrho = fft_conv_same(kernels["Z"], rho)

    H_full = -(
        B[0, 0] * K11rho + B[0, 1] * K12rho + B[1, 0] * K21rho + B[1, 1] * K22rho
    )
    H_tr = -(
        B[0, 0] * K11orho + B[0, 1] * K12orho + B[1, 0] * K21orho + B[1, 1] * K22orho
    )
    H_ct = -(np.trace(B) / 2.0) * Zrho
    return H_full, H_tr, H_ct


def make_fields() -> dict[str, np.ndarray]:
    return {
        "incompressible_hyperbolic": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        "incompressible_rotation": np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float),
        "compressible_identity": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        "weakly_compressible_diag": np.array([[0.6, 0.0], [0.0, -0.2]], dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic structural simulation for the Bressan theorem.")
    parser.add_argument("--out", type=str, default="bressan_sim_v1_out", help="output directory")
    parser.add_argument("--N", type=int, default=301, help="grid size")
    parser.add_argument("--L", type=float, default=10.0, help="half-box size, domain is [-L,L]^2")
    parser.add_argument("--sigma-kernel", type=float, default=0.22, help="base Gaussian eta width")
    parser.add_argument("--eps", type=float, default=0.12, help="multiscale epsilon")
    parser.add_argument("--R", type=float, default=2.0, help="multiscale R")
    parser.add_argument("--nr", type=int, default=80, help="number of quadrature points in r")
    parser.add_argument("--sigma-rho", type=float, default=0.9, help="initial rho bump width")
    parser.add_argument("--rho-cx", type=float, default=1.5, help="initial rho x-center")
    parser.add_argument("--rho-cy", type=float, default=-1.0, help="initial rho y-center")
    parser.add_argument("--T", type=float, default=1.2, help="final time")
    parser.add_argument("--n-time", type=int, default=21, help="number of time slices")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-args.L, args.L, args.N)
    dx = float(x[1] - x[0])
    X, Y = np.meshgrid(x, x, indexing="xy")

    # --------------------------------------------------------
    # Build kernels
    # --------------------------------------------------------
    kernels = build_multiscale_kernels(
        x=x,
        sigma=args.sigma_kernel,
        eps=args.eps,
        R=args.R,
        n_r=args.nr,
    )

    traceK = kernels["K11"] + kernels["K22"]
    traceKo = kernels["K11o"] + kernels["K22o"]
    intZ = integral2d(kernels["Z"], dx)
    log_target = -2.0 * np.log(args.R / args.eps)
    structural = {
        "max_abs_traceK_minus_Z": float(np.max(np.abs(traceK - kernels["Z"]))),
        "max_abs_traceKo": float(np.max(np.abs(traceKo))),
        "int_Z": intZ,
        "minus_2log_R_over_eps": float(log_target),
        "abs_intZ_plus_2log": float(abs(intZ - log_target)),
    }

    # --------------------------------------------------------
    # Dynamic simulation under linear flows
    # --------------------------------------------------------
    fields = make_fields()
    t_grid = np.linspace(0.0, args.T, args.n_time)

    report = {
        "grid": {
            "N": args.N,
            "box_half_size": args.L,
            "dx": dx,
        },
        "kernel_parameters": {
            "sigma_kernel": args.sigma_kernel,
            "eps": args.eps,
            "R": args.R,
            "nr": args.nr,
        },
        "structural_checks": structural,
        "flows": {},
    }

    plt.figure(figsize=(9, 5))
    for name, B in fields.items():
        traceB = float(np.trace(B))
        norms_full = []
        norms_tr = []
        norms_ct = []
        ints_full = []
        ints_ct = []
        recon_errors = []

        # store a middle-time snapshot for plotting
        snapshot_t = t_grid[len(t_grid) // 2]
        snapshot = None

        for t in t_grid:
            rho = rho_transport_linear(
                X, Y, float(t), B,
                cx0=args.rho_cx, cy0=args.rho_cy,
                sigma=args.sigma_rho
            )
            H_full, H_tr, H_ct = dynamic_channels(kernels, rho, B)
            norms_full.append(l2_norm(H_full, dx))
            norms_tr.append(l2_norm(H_tr, dx))
            norms_ct.append(l2_norm(H_ct, dx))
            ints_full.append(integral2d(H_full, dx))
            ints_ct.append(integral2d(H_ct, dx))
            recon_errors.append(float(np.max(np.abs(H_full - (H_tr + H_ct)))))

            if abs(t - snapshot_t) < 1e-12:
                snapshot = {
                    "rho": rho,
                    "H_full": H_full,
                    "H_tr": H_tr,
                    "H_ct": H_ct,
                }

        report["flows"][name] = {
            "trace_B": traceB,
            "max_reconstruction_error": float(np.max(recon_errors)),
            "contact_L2_max": float(np.max(norms_ct)),
            "full_minus_traceless_L2_max": float(np.max(np.abs(np.array(norms_full) - np.array(norms_tr)))),
            "integral_H_full_max_abs": float(np.max(np.abs(ints_full))),
            "integral_H_contact_max_abs": float(np.max(np.abs(ints_ct))),
        }

        plt.plot(t_grid, norms_ct, label=f"{name}: ||H_contact||_2")

        # save snapshot image
        if snapshot is not None:
            fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
            im0 = axs[0].imshow(snapshot["rho"], origin="lower", extent=[x.min(), x.max(), x.min(), x.max()])
            axs[0].set_title(f"{name}\nrho(t={snapshot_t:.2f})")
            plt.colorbar(im0, ax=axs[0], fraction=0.046)

            im1 = axs[1].imshow(snapshot["H_full"], origin="lower", extent=[x.min(), x.max(), x.min(), x.max()])
            axs[1].set_title("H_full")
            plt.colorbar(im1, ax=axs[1], fraction=0.046)

            im2 = axs[2].imshow(snapshot["H_tr"], origin="lower", extent=[x.min(), x.max(), x.min(), x.max()])
            axs[2].set_title("H_traceless")
            plt.colorbar(im2, ax=axs[2], fraction=0.046)

            im3 = axs[3].imshow(snapshot["H_ct"], origin="lower", extent=[x.min(), x.max(), x.min(), x.max()])
            axs[3].set_title("H_contact")
            plt.colorbar(im3, ax=axs[3], fraction=0.046)

            for ax in axs:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            plt.tight_layout()
            fig.savefig(out_dir / f"{name}_snapshot.png", dpi=160)
            plt.close(fig)

    plt.title("Dynamic contact-sector norm under linear flows")
    plt.xlabel("time")
    plt.ylabel(r"$\|H_{\mathrm{contact}}(t)\|_2$")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "contact_norms_vs_time.png", dpi=170)
    plt.close()

    # --------------------------------------------------------
    # Save textual report
    # --------------------------------------------------------
    report_path = out_dir / "bressan_sim_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = []
    txt_lines.append("BRESSAN_SIM_v1 report")
    txt_lines.append("======================")
    txt_lines.append("")
    txt_lines.append(f"grid: N={args.N}, box=[{-args.L},{args.L}]^2, dx={dx:.6f}")
    txt_lines.append(f"kernel parameters: sigma={args.sigma_kernel}, eps={args.eps}, R={args.R}, nr={args.nr}")
    txt_lines.append("")
    txt_lines.append("Structural checks")
    txt_lines.append("-----------------")
    txt_lines.append(f"max |trace(K)-Z|               = {structural['max_abs_traceK_minus_Z']:.3e}")
    txt_lines.append(f"max |trace(K^o)|               = {structural['max_abs_traceKo']:.3e}")
    txt_lines.append(f"int Z                          = {structural['int_Z']:.12f}")
    txt_lines.append(f"-2 log(R/eps)                  = {structural['minus_2log_R_over_eps']:.12f}")
    txt_lines.append(f"|int Z - (-2 log(R/eps))|      = {structural['abs_intZ_plus_2log']:.3e}")
    txt_lines.append("")
    txt_lines.append("Dynamic flow checks")
    txt_lines.append("-------------------")
    for name, data in report["flows"].items():
        txt_lines.append(f"{name}:")
        txt_lines.append(f"  trace(B)                     = {data['trace_B']:.6f}")
        txt_lines.append(f"  max |H_full-(H_tr+H_ct)|     = {data['max_reconstruction_error']:.3e}")
        txt_lines.append(f"  max ||H_contact||_2          = {data['contact_L2_max']:.6e}")
        txt_lines.append(f"  max ||H_full|-|H_tr||        = {data['full_minus_traceless_L2_max']:.6e}")
        txt_lines.append(f"  max |int H_full|             = {data['integral_H_full_max_abs']:.12f}")
        txt_lines.append(f"  max |int H_contact|          = {data['integral_H_contact_max_abs']:.12f}")
        txt_lines.append("")
    (out_dir / "bressan_sim_report.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"Saved output to {out_dir.resolve()}")
    print(f"Main report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
