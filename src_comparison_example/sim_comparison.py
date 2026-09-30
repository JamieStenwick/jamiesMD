"""Run the monodisperse LJ/Langevin comparison; see README.md for units."""
import argparse

from langevinLJsimulation import run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi", type=float, default=0.01)
    parser.add_argument("--N", type=int, default=8000)
    parser.add_argument("--epsilon", type=float, default=5.0, help="Reduced LJ well depth, in the same energy units as kT")
    parser.add_argument("--sigma", type=float, default=1.782, help="LJ length in particle-radius units")
    parser.add_argument("--kT", type=float, default=1.0, help="Dimensionless thermal energy Theta")
    parser.add_argument("--dt", type=float, default=1e-4, help="Dimensionless timestep in units m/gamma")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--time-total", type=int, help="Dimensionless duration, as in jamiesMD (default: 1500)")
    duration.add_argument("--steps", type=int, help="Explicit integration step count instead of --time-total")
    parser.add_argument("--frames", type=int, default=101, help="Saved frames, including initial and final")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-offset", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--outdir", default="hoomd_langevin_comparison")
    parser.add_argument("--no-plot", action="store_true")
    args = vars(parser.parse_args())
    args["plot"] = not args.pop("no_plot")
    run_simulation(**args)


if __name__ == "__main__":
    main()
