"""Overlay total energies from matched jamiesMD and HOOMD simulations.

Run with no arguments to plot the current energies.txt files. Their columns
are potential, kinetic, total; only the third column is plotted, unchanged.
Both runs must use the timestep and frame schedule in HOOMD's parameters.json.
Paths default relative to this script, so it can be run from any directory.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


HERE = Path(__file__).resolve().parent


def load_total_energy(path):
    """Read whole-system total energy, without normalization or smoothing."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 3 or len(data) == 0:
        raise ValueError(f"{path}: expected three columns (potential, kinetic, total)")
    if not np.all(np.isfinite(data[:, 2])):
        raise ValueError(f"{path}: total energy contains nonfinite values")
    return data[:, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hoomd", type=Path, default=HERE / "hoomd_langevin_comparison/energies.txt")
    parser.add_argument("--jamiesmd", type=Path, default=HERE.parent / "jamiesMD/energies.txt")
    parser.add_argument("--parameters", type=Path, default=HERE / "hoomd_langevin_comparison/parameters.json")
    parser.add_argument("--output", type=Path, default=HERE / "total_energy_comparison.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive Matplotlib window")
    args = parser.parse_args()

    hoomd_energy = load_total_energy(args.hoomd)
    jamies_energy = load_total_energy(args.jamiesmd)
    parameters = json.loads(args.parameters.read_text())
    frame_steps = np.asarray(parameters["frame_steps"], dtype=float)
    dt = float(parameters["dt"])
    if frame_steps.ndim != 1 or not np.all(np.isfinite(frame_steps)):
        raise ValueError("parameters.json must contain a finite, one-dimensional frame_steps array")
    if not np.isfinite(dt) or dt <= 0 or np.any(np.diff(frame_steps) <= 0):
        raise ValueError("Require positive dt and strictly increasing frame steps")
    if not len(hoomd_energy) == len(jamies_energy) == len(frame_steps):
        raise ValueError(
            f"Frame counts differ: HOOMD={len(hoomd_energy)}, jamiesMD={len(jamies_energy)}, "
            f"metadata={len(frame_steps)}. Use completed runs with identical frame schedules."
        )
    time = frame_steps * dt

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(time, jamies_energy, color="tab:blue", linewidth=1.6, label="jamiesMD")
    ax.plot(time, hoomd_energy, color="tab:orange", linestyle="--", linewidth=1.6,
            label="HOOMD-blue (Langevin)")
    ax.set_title("Total energy: jamiesMD vs HOOMD-blue")
    ax.set_xlabel(r"Dimensionless time $t^* = t/(m/\gamma)$")
    ax.set_ylabel(r"Total system energy $E^* = U^* + K^*$")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"Saved {args.output.resolve()}")
    print(f"Plotted {len(time)} frames from each run, t*={time[0]:g} to {time[-1]:g}.")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
