"""HOOMD-blue 2.x monodisperse LJ/Langevin comparison with jamiesMD.

All inputs are already dimensionless in jamiesMD units: radius=mass=drag=1,
length unit a, mass unit m_0, energy unit kT_ref, time unit a*sqrt(m_0/E_0).
See README.md for the force shift, inner continuation, and comparison caveats.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import secrets

import numpy as np


RADIUS = MASS = GAMMA = 1.0
TABLE_WIDTH = 50_000


def lennard_jones(r, epsilon, sigma):
    """Unshifted LJ energy and radial force (-dU/dr), in reduced units."""
    r = np.asarray(r, dtype=float)
    s6 = (sigma / r) ** 6
    return 4 * epsilon * s6 * (s6 - 1), 24 * epsilon / r * s6 * (2 * s6 - 1)


def compute_table_bounds(epsilon, sigma, kT):
    """Match getRmin/getRmax: U_LJ(rmin)=-epsilon+20*Theta, rmax=3*sigma.

    The threshold uses the UNshifted LJ well bottom. Float32 input/bound
    rounding follows SimParams/SimTables in jamiesMD.
    """
    epsilon, sigma, kT = (float(np.float32(x)) for x in (epsilon, sigma, kT))
    if not all(math.isfinite(x) for x in (epsilon, sigma, kT)):
        raise ValueError("epsilon, sigma, and kT must be finite")
    if epsilon <= 0 or sigma <= 0 or kT < 0:
        raise ValueError("Require epsilon > 0, sigma > 0, and kT >= 0")
    # Algebraically equivalent to getRmin, without subtractive cancellation.
    y = 0.5 * (1 + math.sqrt(20 * kT / epsilon))
    rmin = float(np.float32(sigma / y ** (1 / 6)))
    rmax = float(np.float32(3 * sigma))
    return rmin, rmax


def shifted_lj(r, rmin, rmax, epsilon, sigma):
    """Force-shifted LJ with jamiesMD's intended continuation below rmin.

    Below rmin, F=F(rmin) and U=U(rmin)+F(rmin)*(rmin-r).
    At/above rmax both are zero. The continuation avoids a zero-force hole
    below HOOMD's main table; it is not an additional hard-sphere potential.
    """
    r = np.asarray(r, dtype=float)
    bounded = np.maximum(r, rmin)
    u, f = lennard_jones(bounded, epsilon, sigma)
    uc, fc = lennard_jones(rmax, epsilon, sigma)
    shifted_force = f - fc
    shifted_energy = u - uc + (bounded - rmax) * fc
    shifted_energy += shifted_force * (bounded - r)
    return np.where(r < rmax, shifted_energy, 0.0), np.where(r < rmax, shifted_force, 0.0)


def frame_schedule(steps, frames):
    """jamiesMD frame indices, including initial/final; C++ round for x>=0."""
    if isinstance(steps, bool) or int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(frames, bool) or int(frames) != frames or not 2 <= frames <= steps + 1:
        raise ValueError("Require 2 <= frames <= steps + 1")
    return np.floor(np.arange(int(frames)) * (int(steps) / (int(frames) - 1)) + 0.5).astype(np.int64)


def generate_positions(N, L, rmin, offset, rng):
    """Keep the existing shuffled cubic-lattice initialization, now seeded."""
    n_side = int(np.ceil(np.cbrt(N)))
    spacing = L / n_side
    if spacing < rmin + offset:
        raise ValueError(f"Lattice spacing {spacing:g} < rmin+offset {rmin + offset:g}; lower phi")
    coords = -L / 2 + (np.arange(n_side) + 0.5) * spacing
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    rng.shuffle(grid)
    return grid[:N]


def _table_sample(r, rmin, rmax, energies, forces):
    """Return a precomputed grid point when HOOMD initializes its table."""
    i = min(len(energies) - 1, max(0, int(round((r - rmin) * (len(energies) - 1) / (rmax - rmin)))))
    return float(energies[i]), float(forces[i])


def run_simulation(
    phi: float,
    epsilon: float,
    sigma: float,
    outdir: str,
    *,
    N: int = 8000,
    dt: float = 1e-4,
    steps: int | None = None,
    time_total: int | None = None,
    frames: int = 301,
    kT: float = 1.0,
    init_offset: float = 0.1,
    device: str = "gpu",
    seed: int | None = None,
    plot: bool = True,
) -> dict:
    """Run HOOMD's underdamped Langevin integrator in jamiesMD units.

    epsilon and kT (Theta) share the reduced energy unit; epsilon is NOT
    multiplied by kT. sigma is in particle-radius units, NOT a radius or
    diameter. Radius, mass, and drag are fixed to 1 to match jamiesMD.
    Initial velocities are Gaussian with std=sqrt(Theta/mass), with the
    center-of-mass velocity subtracted once, as in populateRandom().

    Supply either integer steps or integer dimensionless time_total (default
    1500). The latter uses jamiesMD's int(float32(time_total)/float32(dt)).
    dt itself is rounded to float32 like SimParams, and time=step*dt; no
    Brownian-time or sigma/epsilon-based LJ time rescaling is performed.
    frames includes step zero and the final step. Every frame has a GSD
    snapshot, an energy CSV row, and an energies.txt row (U, K, U+K).
    Energies are whole-system totals, with each pair counted once.

    Requires HOOMD 2.x, NumPy, and (when plot=True) Matplotlib. Use one MPI
    rank, as in run_sim_comparison.sh. Existing output files are overwritten.
    """
    if isinstance(N, bool) or int(N) != N or N < 2:
        raise ValueError("N must be an integer >= 2")
    N = int(N)
    phi, epsilon, sigma, kT, dt = (float(np.float32(x)) for x in (phi, epsilon, sigma, kT, dt))
    if not math.isfinite(phi) or not 0 < phi < 1:
        raise ValueError("Require 0 < phi < 1")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(init_offset) or init_offset < 0:
        raise ValueError("init_offset must be finite and nonnegative")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if steps is not None and time_total is not None:
        raise ValueError("Supply steps or time_total, not both")
    if steps is None:
        time_total = 1500 if time_total is None else time_total
        if not math.isfinite(time_total) or int(time_total) != time_total or time_total <= 0:
            raise ValueError("time_total must be a positive integer, matching jamiesMD's CLI")
        steps = int(np.float32(time_total) / np.float32(dt))
    schedule = frame_schedule(steps, frames)
    steps, frames = int(steps), int(frames)
    rmin, rmax = compute_table_bounds(epsilon, sigma, kT)
    L = float(np.float32((N * (4 * math.pi / 3) * RADIUS**3 / phi) ** (1 / 3)))
    if L <= 2 * rmax:
        raise ValueError("Box length must exceed 2*rmax; increase N or lower phi")
    seed = secrets.randbits(31) if seed is None else seed
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    positions = generate_positions(N, L, rmin, init_offset, rng)
    velocities = rng.normal(0, math.sqrt(kT / MASS), (N, 3))
    velocities -= velocities.mean(axis=0)

    import hoomd
    import hoomd.md

    if not hasattr(hoomd, "context"):
        raise RuntimeError("This comparison targets HOOMD-blue 2.x; use the environment in run_sim_comparison.sh")
    hoomd.context.initialize(f"--mode={device}")
    if hoomd.comm.get_num_ranks() != 1:
        raise RuntimeError("Run this comparison with one MPI rank")
    hoomd.util.quiet_status()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gsd_path": str(outdir / "trajectory.gsd"),
        "energy_csv": str(outdir / "energies.csv"),
        "energy_txt": str(outdir / "energies.txt"),
        "metadata_path": str(outdir / "parameters.json"),
    }
    metadata = dict(phi=phi, epsilon=epsilon, sigma=sigma, kT=kT, radius=RADIUS,
                    mass=MASS, gamma=GAMMA, N=N, dt=dt, steps=steps,
                    requested_time_total=time_total, actual_time_total=steps * dt,
                    frames=frames, frame_steps=schedule.tolist(), box_length=L,
                    rmin=rmin, rmax=rmax, table_width=TABLE_WIDTH,
                    seed=seed, init_offset=init_offset, device=device,
                    potential="force-shifted LJ with linear core", integrator="HOOMD Langevin",
                    time_unit="m/gamma", energy_unit="gamma^2*a^2/m", length_unit="a")
    Path(paths["metadata_path"]).write_text(json.dumps(metadata, indent=2) + "\n")

    snapshot = hoomd.data.make_snapshot(N=N, box=hoomd.data.boxdim(L=L), particle_types=["A"])
    system = hoomd.init.read_snapshot(snapshot)
    # Use particle setters: this HOOMD 2.x build exposes broken zero-stride
    # snapshot NumPy views under NumPy 2, silently corrupting bulk assignment.
    for particle in system.particles:
        particle.position = tuple(positions[particle.tag])
        particle.velocity = tuple(velocities[particle.tag])
        particle.mass = MASS
        particle.diameter = 2 * RADIUS

    nl = hoomd.md.nlist.cell()
    main_table = hoomd.md.pair.table(width=TABLE_WIDTH, nlist=nl, name="lj")
    grid = np.linspace(rmin, rmax, TABLE_WIDTH)
    energies, forces = shifted_lj(grid, rmin, rmax, epsilon, sigma)
    main_table.pair_coeff.set("A", "A", func=_table_sample, rmin=rmin, rmax=rmax,
                              coeff=dict(energies=energies, forces=forces))
    # HOOMD returns zero below a table's lower bound. A separate, exactly
    # linear two-point table supplies jamiesMD's constant-force core there.
    core = hoomd.md.pair.table(width=2, nlist=nl, name="lj_core")
    u0, f0 = float(energies[0]), float(forces[0])
    core.pair_coeff.set("A", "A", func=_table_sample, rmin=0.0, rmax=rmin,
                        coeff=dict(energies=[u0 + f0 * rmin, u0], forces=[f0, f0]))

    group_all = hoomd.group.all()
    hoomd.md.integrate.mode_standard(dt=dt, aniso=False)
    langevin = hoomd.md.integrate.langevin(group=group_all, kT=kT, seed=seed, dscale=False)
    langevin.set_gamma("A", gamma=GAMMA)
    logger = hoomd.analyze.log(filename=None, quantities=["potential_energy", "kinetic_energy"], period=1)
    # Keep the in-memory logger enabled so HOOMD registers its quantities.
    # query() obtains current values rather than a cached file output row.
    hoomd.run(0, quiet=True)

    print(f"LJ/Langevin: N={N}, Theta={kT:g}, dt*={dt:g}, steps={steps}, t*={steps * dt:g}")
    print(f"radius*=mass*=gamma*=1; rmin={rmin:.9g}, rmax={rmax:.9g}; {frames} frames")
    with open(paths["energy_csv"], "w", newline="") as csv_file, open(paths["energy_txt"], "w") as txt_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "time", "potential_energy", "kinetic_energy", "total_energy"])
        for frame, target in enumerate(schedule):
            advance = int(target) - hoomd.get_step()
            if advance:
                hoomd.run(advance, quiet=True)
            u = logger.query("potential_energy")
            k = logger.query("kinetic_energy")
            if not math.isfinite(u + k):
                raise RuntimeError(f"Nonfinite energy at step {target}; try reducing dt")
            hoomd.dump.gsd(filename=paths["gsd_path"], period=None, group=group_all,
                           overwrite=(frame == 0), dynamic=["property", "momentum"])
            writer.writerow([int(target), int(target) * dt, u, k, u + k])
            txt_file.write(f"{u:.16g} {k:.16g} {u + k:.16g}\n")
            csv_file.flush()
            txt_file.flush()
            print(f"Writing frame {frame + 1}/{frames}, step {target}")

    if plot:
        plot_pair_potential(rmin, rmax, epsilon, sigma, outdir / "potential_plot.png")
        plot_energy(paths["energy_csv"], outdir / "energy_plot.png")
    hoomd.util.unquiet_status()
    return {**paths, **metadata}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_pair_potential(rmin, rmax, epsilon, sigma, out_png):
    """Plot the actual shifted LJ interaction and its radial force."""
    plt = _pyplot()
    r = np.linspace(0.9 * rmin, 1.05 * rmax, 2000)
    u, f = shifted_lj(r, rmin, rmax, epsilon, sigma)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(r, u)
    axes[1].plot(r, f)
    for ax in axes:
        ax.axvline(rmin, color="gray", linestyle="--", label="rmin")
        ax.axvline(rmax, color="black", linestyle=":", label="rmax")
        ax.grid(True)
    axes[0].set_ylabel("Pair potential U*")
    axes[0].legend()
    axes[1].set_ylabel("Radial force F*")
    axes[1].set_xlabel("Center separation r/a")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_energy(csv_path, out_png):
    """Plot every saved frame, including t=0, as total system energies."""
    plt = _pyplot()
    data = np.genfromtxt(csv_path, delimiter=",", names=True, ndmin=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, label in [("potential_energy", "Potential"), ("kinetic_energy", "Kinetic"), ("total_energy", "Total")]:
        ax.plot(data["time"], data[name], label=label)
    ax.set_xlabel("Dimensionless time t* = t / (m/gamma)")
    ax.set_ylabel("System energy E*")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
