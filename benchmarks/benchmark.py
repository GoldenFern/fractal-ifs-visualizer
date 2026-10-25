"""Micro-benchmarks for point generation and image export.

Run from the repository root:

    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --n 10000000 --skip-render
"""

import argparse
import os
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import IFS


def _timed(label, func):
    """Run ``func`` once and print its elapsed time."""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    print(f"{label:<34s} {elapsed:8.3f} s")
    return result


def benchmark_generation(n_total):
    """Compare the Python, NumPy and Numba generation backends."""
    print("\nPoint generation")
    print("-" * 46)

    funcs, probs = IFS.define_barnsley_fern()
    for method in ("python", "numpy", "numba"):
        if method == "numba" and not IFS.NUMBA_AVAILABLE:
            continue
        visualizer = IFS.IFSVisualizer(funcs, probs)
        if method == "numba":
            visualizer.generate_points(1000, 10, method="numba", seed=0)
        _timed(
            f"Barnsley fern, {method}",
            lambda v=visualizer, m=method: v.generate_points(
                n_total, 2000, (0.1, 0.1), method=m, seed=0
            ),
        )

    if IFS.NUMBA_AVAILABLE:
        visualizer = IFS.IFSVisualizer(*IFS.my_ifs())
        visualizer.generate_points(1000, 10, method="numba", seed=0)
        _timed(
            "non-affine my_ifs, numba",
            lambda: visualizer.generate_points(
                n_total, 2000, (0.1, 0.1), method="numba", seed=0
            ),
        )


def benchmark_export(n_total):
    """Compare scatter export at several resolutions with a density map."""
    print("\nImage export")
    print("-" * 46)
    visualizer = IFS.IFSVisualizer(*IFS.my_ifs())
    points = visualizer.generate_points(n_total, 2000, (0.1, 0.1), seed=0)

    with tempfile.TemporaryDirectory() as tmp:
        for dpi in (300, 700):
            path = os.path.join(tmp, f"scatter_{dpi}.png")
            _timed(
                f"scatter, {dpi} dpi",
                lambda p=path, d=dpi: visualizer.plot_matplotlib(
                    save=True,
                    filename=p,
                    dpi=d,
                    show=False,
                    close=True,
                ),
            )
            print(f"{'  file size':<34s} {os.path.getsize(path) / 1e6:8.2f} MB")

        for bins in (2000, 4000):
            path = os.path.join(tmp, f"density_{bins}.png")
            _timed(
                f"density, {bins} bins",
                lambda p=path, b=bins: visualizer.plot_matplotlib(
                    save=True,
                    filename=p,
                    mode="density",
                    bins=b,
                    show=False,
                    close=True,
                ),
            )
            print(f"{'  file size':<34s} {os.path.getsize(path) / 1e6:8.2f} MB")

    print(f"\npoint array: {points.nbytes / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000, help="number of iterations")
    parser.add_argument("--skip-render", action="store_true", help="skip image export")
    args = parser.parse_args()

    print(
        f"numpy {np.__version__}, numba "
        f"{'available' if IFS.NUMBA_AVAILABLE else 'not installed'}, n={args.n}"
    )
    benchmark_generation(args.n)
    if not args.skip_render:
        benchmark_export(args.n)


if __name__ == "__main__":
    main()
