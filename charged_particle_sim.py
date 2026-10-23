"""
Charged particle (proton) in a uniform magnetic field:
numerical (SciPy DOP853) vs analytical solution.

Run:  python charged_particle_sim.py
Outputs: printed metrics, convergence table, and figures (PNG).
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- physical parameters (Table I) ----------
q  = 1.602176634e-19      # C
m  = 1.67262192369e-27    # kg
B0 = 0.5                  # T, field along +z
r0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([2.0e5, 0.0, 1.0e5])   # m/s
N_SAMPLES = 3000
N_PERIODS = 5

Omega = q * B0 / m                    # signed angular frequency (rad/s)
T = 2 * np.pi / abs(Omega)            # cyclotron period (s)
t_end = N_PERIODS * T
v_perp = np.hypot(v0[0], v0[1])
r_L = m * v_perp / (abs(q) * B0)      # gyration radius (m)
pitch = v0[2] * T                     # helix pitch (m)


def rhs(t, Y):
    x, y, z, vx, vy, vz = Y
    return [vx, vy, vz, Omega * vy, -Omega * vx, 0.0]


def analytical(t):
    vx0, vy0, vz0 = v0
    c, s = np.cos(Omega * t), np.sin(Omega * t)
    x = r0[0] + vx0 / Omega * s + vy0 / Omega * (1 - c)
    y = r0[1] + vy0 / Omega * s - vx0 / Omega * (1 - c)
    z = r0[2] + vz0 * t
    vx = vx0 * c + vy0 * s
    vy = vy0 * c - vx0 * s
    vz = vz0 * np.ones_like(t)
    return np.vstack([x, y, z]), np.vstack([vx, vy, vz])


def run(rtol, atol, n=N_SAMPLES):
    t = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs, (0.0, t_end), np.concatenate([r0, v0]),
                    method="DOP853", t_eval=t, rtol=rtol, atol=atol)
    r_num, v_num = sol.y[:3], sol.y[3:]
    r_ana, _ = analytical(t)
    err = np.linalg.norm(r_num - r_ana, axis=0)
    K = 0.5 * m * np.sum(v_num ** 2, axis=0)
    dK_rel = np.max(np.abs(K - K[0]) / K[0])
    dK_abs = np.max(np.abs(K - K[0]))
    return dict(t=t, r_num=r_num, r_ana=r_ana, err=err, K0=K[0],
                dK_rel=dK_rel, dK_abs=dK_abs, nfev=sol.nfev)


if __name__ == "__main__":
    res = run(1e-10, 1e-12)
    print(f"Omega   = {Omega:.6e} rad/s")
    print(f"T       = {T:.6e} s")
    print(f"t_end   = {t_end:.6e} s")
    print(f"r_L     = {r_L:.6e} m")
    print(f"pitch   = {pitch:.6e} m")
    print(f"e_max   = {res['err'].max():.6e} m")
    print(f"K0      = {res['K0']:.6e} J")
    print(f"dK_rel  = {res['dK_rel']:.6e}")
    print(f"dK_abs  = {res['dK_abs']:.6e} J")

    # ---- convergence study over solver tolerances ----
    print("\nConvergence study (atol = rtol/100):")
    print(f"{'rtol':>8} {'atol':>8} {'e_max (m)':>14} {'dK_rel':>14} {'nfev':>7}")
    rows = []
    for k in (6, 7, 8, 9, 10, 11, 12):
        rt, at = 10.0 ** -k, 10.0 ** -(k + 2)
        r = run(rt, at)
        rows.append((rt, at, r["err"].max(), r["dK_rel"], r["nfev"]))
        print(f"{rt:8.0e} {at:8.0e} {r['err'].max():14.6e} {r['dK_rel']:14.6e} {r['nfev']:7d}")
    np.save("convergence.npy", np.array(rows))

    # ---- figures ----
    t, rn, ra, err = res["t"], res["r_num"], res["r_ana"], res["err"]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rn[0] * 1e3, rn[1] * 1e3, rn[2] * 1e3, lw=1.2)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)", labelpad=8)
    ax.set_title("Three-dimensional particle trajectory")
    fig.tight_layout(); fig.savefig("fig1_3d.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(rn[0] * 1e3, rn[1] * 1e3, lw=2.5, label="Numerical (DOP853)")
    ax.plot(ra[0] * 1e3, ra[1] * 1e3, "--", lw=1.2, label="Analytical")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_aspect("equal")
    ax.set_title("Trajectory in the xy-plane"); ax.grid(alpha=.4)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig("fig2_xy.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(t[1:] * 1e7, np.maximum(err[1:], 1e-18), lw=0.9)
    ax.set_ylim(1e-16, 1e-11)
    ax.set_xlabel(r"Time ($\times 10^{-7}$ s)"); ax.set_ylabel("Position error (m)")
    ax.set_title("Absolute position error (numerical vs analytical)")
    ax.grid(alpha=.4, which="both")
    fig.tight_layout(); fig.savefig("fig3_error.png", dpi=200); plt.close(fig)
    print("figures saved")
