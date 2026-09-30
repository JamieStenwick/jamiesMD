"""Compare saxs-fft S(q) from HOOMD and jamiesMD, inside a Slurm allocation.

Submit run_structure_factor_comparison.sh. By default, average the last 100
frames from each run. Input paths are relative to this script, not the caller.
The runs must share N, box, timestep, and frame schedule in parameters.json.

The converted jamiesMD GSD keeps radius=1 / diameter=2 units. For BOTH FFTs,
positions AND box lengths are divided by 2 to satisfy saxs-fft's diameter=1
convention. The returned q*d is divided by 2 for plotting q*a. S(q) is not
rescaled; no form factor, contrast, physical diameter, or intensity is used.
"""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SAXS_REPO = HERE.parent / "characterization" / "saxs-fft"
DIAMETER = 2.0


def convert_positions_to_gsd(positions_path, output_path, parameters):
    """Convert N rows per frame to a position-only GSD in native radius units.

    positions.txt has no box or time metadata, so those come from the matched
    HOOMD run. Velocities cannot be recovered and are omitted. Periodic image
    indices preserve unwrapped coordinates to the precision of the text input.
    """
    import gsd.hoomd
    import numpy as np

    n = int(parameters["N"])
    steps = np.asarray(parameters["frame_steps"], dtype=np.int64)
    length = float(parameters["box_length"])
    if n < 1 or not np.isfinite(length) or length <= 0:
        raise ValueError("Require positive N and finite positive box length")
    if steps.ndim != 1 or len(steps) < 1 or np.any(steps < 0) or np.any(np.diff(steps) <= 0):
        raise ValueError("Frame steps must be nonnegative and strictly increasing")
    positions = np.loadtxt(positions_path, ndmin=2)
    if positions.shape != (len(steps) * n, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"{positions_path}: expected {len(steps) * n} finite x,y,z rows "
            f"({len(steps)} frames x {n} particles); got {positions.shape}"
        )
    if output_path.resolve() == positions_path.resolve():
        raise ValueError("Converted GSD must not overwrite source text")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.gsd")
    with gsd.hoomd.open(str(temporary), "w") as trajectory:
        for index, xyz in enumerate(positions.reshape(-1, n, 3)):
            frame = gsd.hoomd.Frame()
            frame.configuration.step = int(steps[index])
            frame.configuration.dimensions = 3
            frame.configuration.box = [length, length, length, 0, 0, 0]
            frame.particles.N = n
            frame.particles.types = ["A"]
            frame.particles.diameter = np.full(n, DIAMETER, dtype=np.float32)
            images = np.floor((xyz + length / 2) / length).astype(np.int32)
            wrapped = (xyz - images * length).astype(np.float32)
            # Float32 rounding can put a positive boundary point at +L/2.
            overflow = wrapped >= np.float32(length / 2)
            wrapped[overflow] -= np.float32(length)
            images[overflow] += 1
            frame.particles.position = wrapped
            frame.particles.image = images
            trajectory.append(frame)
    temporary.replace(output_path)
    print(f"Converted {len(steps)} frames to {output_path}", flush=True)


def average_structure_factor(path, parameters, last_frames, grid, trim_bins, device):
    """Call the local repo's FFT kernel directly, avoiding its text round trip.

    Sum shell numerators and counts exactly as StructureFactor.compute_s_1d
    does. With fixed N and box, this also gives equal weight to each frame.
    """
    import gsd.hoomd
    import numpy as np
    import torch
    from saxsfft.structurefactor import compute_s_1d

    expected_steps = np.asarray(parameters["frame_steps"], dtype=np.int64)
    length = float(parameters["box_length"])
    trim = slice(trim_bins, -trim_bins) if trim_bins else slice(None)
    num_total = cnt_total = q_ref = None
    with gsd.hoomd.open(str(path), "r") as trajectory:
        if len(trajectory) != len(expected_steps) or len(trajectory) == 0:
            raise ValueError(f"{path}: trajectory frame count differs from metadata or is empty")
        indices = list(range(max(0, len(trajectory) - last_frames), len(trajectory)))
        for count, index in enumerate(indices, 1):
            frame = trajectory[index]
            box = np.asarray(frame.configuration.box, dtype=float)
            if frame.particles.N != parameters["N"] or frame.configuration.dimensions != 3:
                raise ValueError(f"{path}, frame {index}: N or dimensionality differs from the matched 3D run")
            if frame.configuration.step != expected_steps[index]:
                raise ValueError(f"{path}, frame {index}: timestep differs from metadata")
            if not np.allclose(box[:3], length, rtol=1e-7, atol=0) or np.any(box[3:] != 0):
                raise ValueError("Require matching fixed cubic, untilted boxes")
            if not np.allclose(frame.particles.diameter, DIAMETER):
                raise ValueError(f"{path}: expected monodisperse diameter=2 (radius=1)")
            xyz = np.asarray(frame.particles.position, dtype=float)
            if not np.all(np.isfinite(xyz)):
                raise ValueError(f"{path}, frame {index}: nonfinite positions")
            q, numerator, counts = compute_s_1d(
                xyz / DIAMETER, box[:3] / DIAMETER, N_grid=grid,
                particle_diameter=None, trim=trim, device=device, dtype=torch.float64,
            )
            if q_ref is None:
                q_ref = q
                num_total = np.zeros_like(numerator)
                cnt_total = np.zeros_like(counts)
            elif q.shape != q_ref.shape or not np.allclose(q, q_ref, rtol=1e-12, atol=0):
                raise ValueError("Wavevector grids differ across frames")
            num_total += numerator
            cnt_total += counts
            if count == 1 or count % 10 == 0 or count == len(indices):
                print(f"{path.name}: FFT frame {count}/{len(indices)} (index {index})", flush=True)

    # Keep complete spherical shells within the axial Nyquist limit, excluding
    # the incomplete corner shells also returned by the package.
    length_diameter = length / DIAMETER
    dq = 2 * np.pi / length_diameter
    q_nyquist = np.pi * grid / length_diameter
    valid = (cnt_total > 0) & (q_ref > 0) & (q_ref + dq / 2 <= q_nyquist)
    if not np.any(valid):
        raise ValueError("No usable q bins; increase --grid or decrease --trim-bins")
    sq = num_total[valid] / cnt_total[valid]
    if not np.all(np.isfinite(sq)):
        raise ValueError("Nonfinite structure factor")
    return q_ref[valid] / DIAMETER, sq, indices


def plot_comparison(q_radius, jamies_sq, hoomd_sq, frames, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(q_radius, jamies_sq, color="tab:blue", linewidth=1.7, label="jamiesMD")
    ax.loglog(q_radius, hoomd_sq, color="tab:orange", linestyle="--", linewidth=1.7,
              label="HOOMD-blue (Langevin)")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":", label="S(q) = 1")
    ax.set_xlabel(r"Wavevector $q a$ (particle radius $a=1$)")
    ax.set_ylabel(r"Structure factor $S(q)$ (dimensionless)")
    ax.set_title(f"Structure factor — average over the last {frames} frames")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hoomd", type=Path, default=HERE / "hoomd_langevin_comparison/trajectory.gsd")
    parser.add_argument("--positions", type=Path, default=HERE.parent / "jamiesMD/positions.txt")
    parser.add_argument("--parameters", type=Path, default=HERE / "hoomd_langevin_comparison/parameters.json")
    parser.add_argument("--outdir", type=Path, default=HERE / "structure_factor_comparison")
    parser.add_argument("--last-frames", type=int, default=100)
    parser.add_argument("--grid", type=int, default=256, help="FFT points per box side (default: 256)")
    parser.add_argument("--trim-bins", type=int, default=3, help="saxs-fft edge trimming (default: 3)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=4, help="Torch CPU threads (default: 4)")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run inside a Slurm allocation: sbatch run_structure_factor_comparison.sh")
    if args.last_frames < 1 or args.grid < 16 or args.trim_bins < 0 or args.threads < 1:
        parser.error("Require --last-frames >= 1, --grid >= 16, --trim-bins >= 0, --threads >= 1")
    if args.threads > int(os.environ.get("SLURM_CPUS_PER_TASK", args.threads)):
        parser.error("--threads exceeds SLURM_CPUS_PER_TASK")

    # Import the numerical stack only after the allocation check.
    import numpy as np
    import torch
    sys.path.insert(0, str(SAXS_REPO))
    import saxsfft
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; request a GPU or use --device cpu inside a compute allocation")
    torch.set_num_threads(args.threads)
    parameters = json.loads(args.parameters.read_text())
    if float(parameters.get("radius", 1)) != 1:
        parser.error("This script expects both simulations to use radius=1")
    args.outdir.mkdir(parents=True, exist_ok=True)
    converted = args.outdir / "jamiesMD_positions.gsd"
    if converted.resolve() == args.hoomd.resolve():
        parser.error("Converted GSD must not overwrite the HOOMD input")
    convert_positions_to_gsd(args.positions, converted, parameters)
    print(f"saxs-fft: {saxsfft.__file__}; device={args.device}, grid={args.grid}^3", flush=True)
    q_h, s_h, frames_h = average_structure_factor(
        args.hoomd, parameters, args.last_frames, args.grid, args.trim_bins, args.device)
    q_j, s_j, frames_j = average_structure_factor(
        converted, parameters, args.last_frames, args.grid, args.trim_bins, args.device)
    np.testing.assert_allclose(q_h, q_j, rtol=1e-12, atol=0)
    if frames_h != frames_j:
        raise ValueError("The frame windows do not match")
    np.savetxt(args.outdir / "structure_factors.txt", np.column_stack([q_h, 2*q_h, s_j, s_h]),
               header="q_times_radius q_times_diameter S_jamiesMD S_HOOMD", fmt="%.10e")
    np.savez(args.outdir / "structure_factors.npz", q_radius=q_h, q_diameter=2*q_h,
             S_jamiesMD=s_j, S_HOOMD=s_h)
    plot_path = args.outdir / "structure_factor_comparison.png"
    plot_comparison(q_h, s_j, s_h, len(frames_h), plot_path)
    analysis = dict(
        hoomd_gsd=str(args.hoomd.resolve()), jamiesmd_positions=str(args.positions.resolve()),
        converted_gsd=str(converted.resolve()), saxs_module=saxsfft.__file__,
        saxs_version=saxsfft.__version__, slurm_job_id=os.environ["SLURM_JOB_ID"],
        frame_indices=frames_h, frame_count=len(frames_h), requested_last_frames=args.last_frames,
        first_step=parameters["frame_steps"][frames_h[0]], last_step=parameters["frame_steps"][frames_h[-1]],
        first_time=parameters["dt"] * parameters["frame_steps"][frames_h[0]],
        last_time=parameters["dt"] * parameters["frame_steps"][frames_h[-1]],
        N=parameters["N"], box_length_radius_units=parameters["box_length"],
        grid=args.grid, trim_bins=args.trim_bins, dtype="float64", device=args.device,
        cpu_threads=args.threads, input_diameter=DIAMETER,
        fft_length_unit="particle diameter d=2a", plot_q="q*a = (q*d)/2",
        q_radius_min=float(q_h[0]), q_radius_max=float(q_h[-1]),
        averaging="sum of shell numerators / sum of shell counts",
        converted_gsd_note="Position-only analysis trajectory; no measured velocities in positions.txt")
    (args.outdir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"Saved {plot_path}; averaged {len(frames_h)} frames per simulation", flush=True)


if __name__ == "__main__":
    main()
