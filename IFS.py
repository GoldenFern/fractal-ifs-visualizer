"""
分形几何可视化：IFS系统。
目前先在二维平面上进行操作。
可视化方法：
1. 用户构造IFS系统的压缩映射函数；
2. 确定分形所在的集合的范围；
3. 确定初始点的选择（默认随机选择一个点）；
4. 通过构造迭代序列，剔除前m个点，总共n个点，作为分形的近似绘制。

点集生成：
- 默认使用 Numba 编译的单条混沌游戏轨道；
- 未安装 Numba 时，仿射系统回退到 NumPy 多链批量实现；
- 其他情况回退到纯 Python 循环。

绘制方法的选择：
1. 基于matplotlib绘制（散点图或密度图）
2. 基于plotly绘制
可选是否保存。matplotlib保存为png/jpg格式，plotly保存为html格式。
"""

import json
import math
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    njit = None
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _numba_affine_chaos_game(
        A: np.ndarray,
        b: np.ndarray,
        cdf: np.ndarray,
        n_total: int,
        m_discard: int,
        x0: float,
        y0: float,
        seed: int,
    ) -> np.ndarray:
        """Compiled chaos game for an affine IFS (single trajectory)."""
        n_keep = n_total - m_discard
        points = np.empty((n_keep, 2), dtype=np.float64)
        np.random.seed(seed)

        x = x0
        y = y0
        kept = 0
        n_maps = A.shape[0]

        for i in range(n_total):
            r = np.random.random()
            j = 0
            while j < n_maps - 1 and cdf[j] < r:
                j += 1

            xn = A[j, 0, 0] * x + A[j, 0, 1] * y + b[j, 0]
            yn = A[j, 1, 0] * x + A[j, 1, 1] * y + b[j, 1]
            x = xn
            y = yn

            if i >= m_discard:
                points[kept, 0] = x
                points[kept, 1] = y
                kept += 1

        return points

    @njit(cache=True)
    def _numba_chaos_game(
        step: Callable[[float, float, int], Tuple[float, float]],
        cdf: np.ndarray,
        n_total: int,
        m_discard: int,
        x0: float,
        y0: float,
        seed: int,
    ) -> np.ndarray:
        """Compiled chaos game for arbitrary compiled step functions."""
        n_keep = n_total - m_discard
        points = np.empty((n_keep, 2), dtype=np.float64)
        np.random.seed(seed)

        x = x0
        y = y0
        kept = 0
        n_maps = cdf.shape[0]

        for i in range(n_total):
            r = np.random.random()
            j = 0
            while j < n_maps - 1 and cdf[j] < r:
                j += 1

            x, y = step(x, y, j)

            if i >= m_discard:
                points[kept, 0] = x
                points[kept, 1] = y
                kept += 1

        return points

else:  # pragma: no cover - depends on the environment
    _numba_affine_chaos_game = None
    _numba_chaos_game = None


def _probability_cdf(probabilities: Sequence[float]) -> np.ndarray:
    """Return a contiguous cumulative distribution with a hard upper bound of 1."""
    cdf = np.cumsum(np.asarray(probabilities, dtype=np.float64))
    cdf[-1] = 1.0
    return np.ascontiguousarray(cdf)


def _try_extract_affine(
    functions: Sequence[Callable[[float, float], Tuple[float, float]]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(A, b)`` when every mapping is affine, otherwise ``None``.

    The check evaluates each mapping at a few probe points and verifies the
    linear relationship on two additional points. Nonlinear maps such as the
    sine term in :func:`my_ifs` fail this test and use the compiled dispatcher
    or the Python loop instead.
    """
    probes = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (2.3, -1.7),
        (-0.4, 3.1),
    )
    n_maps = len(functions)
    A = np.empty((n_maps, 2, 2), dtype=np.float64)
    b = np.empty((n_maps, 2), dtype=np.float64)

    for i, func in enumerate(functions):
        try:
            values = [
                np.asarray(func(x, y), dtype=np.float64).reshape(2)
                for x, y in probes
            ]
        except Exception:
            return None

        f0, fx, fy, p, q = values
        a0 = fx - f0
        a1 = fy - f0

        expected_p = f0 + a0 * probes[3][0] + a1 * probes[3][1]
        expected_q = f0 + a0 * probes[4][0] + a1 * probes[4][1]
        if not np.allclose(expected_p, p, rtol=1e-9, atol=1e-10):
            return None
        if not np.allclose(expected_q, q, rtol=1e-9, atol=1e-10):
            return None

        A[i, :, 0] = a0
        A[i, :, 1] = a1
        b[i] = f0

    return A, b


def _compile_numba_step(
    functions: Sequence[Callable[[float, float], Tuple[float, float]]],
) -> Callable[[float, float, int], Tuple[float, float]]:
    """Compile the mappings and a small integer dispatcher with Numba.

    Raises:
        Exception: If Numba cannot compile one of the mappings. The caller
            decides whether to fall back to another backend.
    """
    compiled = []
    for func in functions:
        compiled_func = func if hasattr(func, "py_func") else njit(func)
        sample = compiled_func(0.1, 0.2)
        np.asarray(sample, dtype=np.float64).reshape(2)
        compiled.append(compiled_func)

    # Numba cannot index a heterogeneous tuple of functions with a runtime
    # index, so generate a small if-chain dispatcher instead.
    lines = ["def _dispatcher(x, y, k):"]
    for i in range(len(compiled)):
        lines.append(f"    if k == {i}:")
        lines.append(f"        return f{i}(x, y)")
    lines.append("    return x, y")

    namespace = {f"f{i}": func for i, func in enumerate(compiled)}
    exec(compile("\n".join(lines), "<ifs_dispatcher>", "exec"), namespace)
    dispatcher = njit(namespace["_dispatcher"])
    dispatcher(0.1, 0.2, 0)
    return dispatcher


def _numpy_burn_in(A: np.ndarray, m_discard: int) -> int:
    """Choose a per-chain burn-in from the affine contraction factor."""
    if m_discard <= 0:
        return 0
    factor = float(max(np.linalg.norm(matrix, 2) for matrix in A))
    if 0.0 < factor < 1.0:
        needed = int(np.ceil(np.log(1e-9) / np.log(factor)))
        return max(1, min(m_discard, needed))
    return m_discard


class IFSVisualizer:
    """
    Iterated Function System (IFS) Visualizer for 2D Fractal Geometry.

    Attributes:
        functions (list): A list of callable functions representing contraction mappings.
                          Each function should take (x, y) and return (new_x, new_y).
        probabilities (list): A list of probabilities corresponding to each function.
                              Must sum to 1.
        points (np.ndarray): Generated points (x, y) after iteration.
        backend (str): Backend used by the most recent call to
            :meth:`generate_points`.
    """

    def __init__(
        self,
        functions: List[Callable[[float, float], Tuple[float, float]]],
        probabilities: Optional[Sequence[float]] = None,
        *,
        affine: Optional[Tuple[Any, Any]] = None,
    ):
        """
        Args:
            functions: Contraction mappings, each mapping ``(x, y)`` to
                ``(x', y')``.
            probabilities: Selection probabilities, one per mapping. Uniform
                probabilities are used when omitted.
            affine: Optional precomputed affine coefficients ``(A, b)`` with
                shapes ``(n, 2, 2)`` and ``(n, 2)``. When omitted, affine maps
                are detected lazily on the first call to
                :meth:`generate_points`.
        """
        if not functions:
            raise ValueError("At least one contraction mapping is required.")
        if not all(callable(func) for func in functions):
            raise TypeError("Every contraction mapping must be callable.")
        self.functions = list(functions)

        # If no probabilities provided, assume uniform distribution
        if probabilities is None:
            n = len(self.functions)
            self.probabilities = [1.0 / n] * n
        else:
            self.probabilities = [float(p) for p in probabilities]

        if len(self.probabilities) != len(self.functions):
            raise ValueError("probabilities must have the same length as functions.")
        if any(p < 0.0 for p in self.probabilities):
            raise ValueError("probabilities must be non-negative.")
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.")

        self.points: Optional[np.ndarray] = None
        self.backend: Optional[str] = None

        self._affine: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._affine_checked = False
        if affine is not None:
            A = np.ascontiguousarray(affine[0], dtype=np.float64)
            b = np.ascontiguousarray(affine[1], dtype=np.float64)
            if A.shape != (len(self.functions), 2, 2) or b.shape != (len(self.functions), 2):
                raise ValueError("affine must have shapes (n, 2, 2) and (n, 2).")
            self._affine = (A, b)
            self._affine_checked = True

        self._numba_step: Optional[Callable[[float, float, int], Tuple[float, float]]] = None
        self._numba_step_failed = False

    def _ensure_affine(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect affine mappings once and cache the result."""
        if not self._affine_checked:
            self._affine = _try_extract_affine(self.functions)
            self._affine_checked = True
        return self._affine

    def generate_points(
        self,
        n_total: int,
        m_discard: int,
        initial_point: Tuple[float, float] = (0.0, 0.0),
        method: str = "auto",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates fractal points using the Chaos Game algorithm.

        Args:
            n_total: Total number of iterations.
            m_discard: Number of initial iterations to discard (transient
                state).
            initial_point: Starting coordinate ``(x, y)``.
            method: One of ``"auto"``, ``"numba"``, ``"numpy"`` or
                ``"python"``. ``"auto"`` uses Numba when installed, then a
                vectorized NumPy implementation for affine maps, and finally
                the pure Python loop.
            seed: Seed for the random number generator. The Numba and NumPy
                backends use different generators, so the same seed produces
                reproducible but not identical sequences across backends.

        Returns:
            An array of shape ``(n_total - m_discard, 2)``.

        Raises:
            ValueError: If the arguments are invalid.
            RuntimeError: If an explicitly requested backend is unavailable
                or fails.
        """
        if n_total <= 0:
            raise ValueError("n_total must be positive.")
        if m_discard < 0:
            raise ValueError("m_discard must be non-negative.")
        if n_total <= m_discard:
            raise ValueError("n_total must be greater than m_discard")
        if method not in {"auto", "numba", "numpy", "python"}:
            raise ValueError("method must be 'auto', 'numba', 'numpy' or 'python'.")
        if method == "numba" and not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is not installed; install numba or use method='auto'.")

        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        x0, y0 = float(initial_point[0]), float(initial_point[1])
        affine = self._ensure_affine()

        backend = method
        if backend == "auto":
            if NUMBA_AVAILABLE:
                backend = "numba"
            else:
                backend = "numpy" if affine is not None else "python"

        points: Optional[np.ndarray] = None
        if backend == "numba":
            try:
                points = self._generate_numba(n_total, m_discard, x0, y0, seed, affine)
            except Exception as exc:
                if method == "numba":
                    raise RuntimeError("Numba generation failed.") from exc
                warnings.warn(
                    f"Numba generation failed ({exc}); falling back to a slower backend.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                backend = "numpy" if affine is not None else "python"

        if points is None and backend == "numpy":
            if affine is None:
                if method == "numpy":
                    raise RuntimeError("The NumPy backend requires affine maps.")
                warnings.warn(
                    "The NumPy backend requires affine maps; falling back to the Python loop.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                backend = "python"
            else:
                points = self._generate_numpy(n_total, m_discard, x0, y0, seed, affine)

        if points is None:
            backend = "python"
            points = self._generate_python(n_total, m_discard, x0, y0, seed)

        self.backend = backend
        self.points = points
        print(f"Generated {len(points)} points after discarding first {m_discard} ({backend}).")
        return points

    def _generate_numba(
        self,
        n_total: int,
        m_discard: int,
        x0: float,
        y0: float,
        seed: int,
        affine: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Generate points with a compiled single trajectory."""
        cdf = _probability_cdf(self.probabilities)
        if affine is not None:
            A, b = affine
            return _numba_affine_chaos_game(A, b, cdf, n_total, m_discard, x0, y0, seed)

        if self._numba_step is None:
            if self._numba_step_failed:
                raise RuntimeError("Numba compilation previously failed.")
            try:
                self._numba_step = _compile_numba_step(self.functions)
            except Exception:
                self._numba_step_failed = True
                raise
        return _numba_chaos_game(self._numba_step, cdf, n_total, m_discard, x0, y0, seed)

    def _generate_numpy(
        self,
        n_total: int,
        m_discard: int,
        x0: float,
        y0: float,
        seed: int,
        affine: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Generate points with multiple independent chains in NumPy.

        Each chain is a valid chaos game trajectory, so the union samples the
        same invariant measure as a single long trajectory.
        """
        A, b = affine
        rng = np.random.default_rng(seed)
        cdf = _probability_cdf(self.probabilities)
        n_keep = n_total - m_discard
        n_walkers = min(8192, max(1, n_keep))
        steps = int(np.ceil(n_keep / n_walkers))
        burn_in = _numpy_burn_in(A, m_discard)

        x = np.full(n_walkers, x0, dtype=np.float64)
        y = np.full(n_walkers, y0, dtype=np.float64)
        out = np.empty((steps, n_walkers, 2), dtype=np.float64)

        A00 = np.ascontiguousarray(A[:, 0, 0])
        A01 = np.ascontiguousarray(A[:, 0, 1])
        A10 = np.ascontiguousarray(A[:, 1, 0])
        A11 = np.ascontiguousarray(A[:, 1, 1])
        b0 = np.ascontiguousarray(b[:, 0])
        b1 = np.ascontiguousarray(b[:, 1])

        for step in range(steps + burn_in):
            idx = np.searchsorted(cdf, rng.random(n_walkers), side="right")
            xn = A00[idx] * x + A01[idx] * y + b0[idx]
            yn = A10[idx] * x + A11[idx] * y + b1[idx]
            x, y = xn, yn
            if step >= burn_in:
                out[step - burn_in, :, 0] = x
                out[step - burn_in, :, 1] = y

        return out.reshape(-1, 2)[:n_keep]

    def _generate_python(
        self,
        n_total: int,
        m_discard: int,
        x0: float,
        y0: float,
        seed: int,
    ) -> np.ndarray:
        """Generate points with the pure Python fallback."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            len(self.functions),
            size=n_total,
            p=self.probabilities,
        ).tolist()

        x, y = x0, y0
        x_list: List[float] = []
        y_list: List[float] = []
        append_x = x_list.append
        append_y = y_list.append
        funcs = self.functions

        for i, idx in enumerate(indices):
            x, y = funcs[idx](x, y)
            if i >= m_discard:
                append_x(x)
                append_y(y)

        return np.column_stack((x_list, y_list))

    def plot_matplotlib(
        self,
        save: bool = False,
        filename: str = "ifs_fractal.jpg",
        color: Any = "green",
        alpha: float = 0.4,
        mode: str = "scatter",
        bins: int = 2000,
        cmap: str = "inferno",
        dpi: int = 700,
        show: bool = True,
        close: bool = False,
        figsize: float = 15.0,
    ) -> Tuple[Any, Any]:
        """
        Visualizes the fractal using Matplotlib (Static).

        Args:
            save: Whether to write the figure to disk.
            filename: Output path. The format is inferred from the extension.
            color: Marker color in scatter mode.
            alpha: Marker opacity in scatter mode.
            mode: ``"scatter"`` for the original point cloud or ``"density"``
                for a log-scaled 2D histogram. Density mode is much faster and
                produces much smaller files for large point sets.
            bins: Histogram bins per axis in density mode.
            cmap: Colormap for density mode.
            dpi: Output resolution in scatter mode. In density mode the
                effective dpi is capped so that the saved image is at most
                ``bins`` pixels wide; a larger canvas adds no information.
            show: Whether to open an interactive window.
            close: Whether to close the figure before returning.
            figsize: Figure side length in inches.

        Returns:
            The Matplotlib ``(figure, axes)`` pair.
        """
        if self.points is None:
            raise RuntimeError("No points generated. Please run generate_points() first.")
        if mode not in {"scatter", "density"}:
            raise ValueError("mode must be 'scatter' or 'density'.")

        fig, ax = plt.subplots(figsize=(figsize, figsize))

        if mode == "scatter":
            ax.scatter(self.points[:, 0], self.points[:, 1], s=0.2, color=color, alpha=alpha)
            # ``datalim`` matches the original ``plt.axis("equal")`` behaviour:
            # the square axes are kept and the data limits are expanded.
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            save_dpi = dpi
        else:
            histogram, x_edges, y_edges = np.histogram2d(
                self.points[:, 0],
                self.points[:, 1],
                bins=bins,
            )
            density = np.log1p(histogram.T)
            vmax = float(density.max()) if density.size and density.max() > 0 else 1.0
            ax.imshow(
                density,
                origin="lower",
                extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                cmap=cmap,
                interpolation="nearest",
                aspect="equal",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_axis_off()
            # Let the axes fill the figure so the saved image can reach the
            # full histogram resolution instead of the default subplot area.
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
            save_dpi = min(float(dpi), max(1.0, bins / figsize))

        if save:
            if mode == "density":
                fig.savefig(filename, dpi=save_dpi, bbox_inches="tight", pad_inches=0.0)
            else:
                fig.savefig(filename, dpi=save_dpi, bbox_inches="tight")
            print(f"Image saved to {os.path.abspath(filename)}")

        if show:
            plt.show()
        if close:
            plt.close(fig)
        return fig, ax

    def plot_plotly(
        self,
        save: bool = False,
        filename: str = "ifs_fractal.html",
        color: Any = "green",
        max_points: Optional[int] = None,
        include_plotlyjs: Any = "auto",
        show: bool = True,
    ) -> Any:
        """
        Visualizes the fractal using Plotly (Interactive).

        Uses WebGL (Scattergl) for performance with large datasets.

        Args:
            save: Whether to write the HTML file.
            filename: Output HTML path.
            color: Marker color.
            max_points: Optional cap on the number of plotted points. A value
                such as ``200_000`` keeps the HTML file small and interactive.
            include_plotlyjs: Passed to ``write_html``. Use ``"cdn"`` to avoid
                embedding the Plotly JavaScript bundle.
            show: Whether to open the interactive view.

        Returns:
            The Plotly figure.
        """
        if self.points is None:
            raise RuntimeError("No points generated. Please run generate_points() first.")
        if max_points is not None and max_points <= 0:
            raise ValueError("max_points must be positive or None.")

        points = self.points
        if max_points is not None and len(points) > max_points:
            stride = int(np.ceil(len(points) / max_points))
            points = points[::stride]
            print(f"Plotting {len(points)} of {len(self.points)} points (stride {stride}).")

        fig = go.Figure(data=go.Scattergl(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            marker=dict(
                size=1.5,
                color=color,
                opacity=0.6,
            ),
            name="Fractal Points",
        ))

        fig.update_layout(
            title=r"$\text{Interactive IFS Fractal}$",
            xaxis_title=r"$x$",
            yaxis_title=r"$y$",
            template="plotly_white",
            width=900,
            height=900,
            showlegend=False,
        )

        # Fix aspect ratio
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        if save:
            fig.write_html(filename, include_plotlyjs=include_plotlyjs)
            print(f"Interactive plot saved to {os.path.abspath(filename)}")

        if show:
            fig.show()
        return fig


# ==========================================
# Common Fractal IFS Definitions (保留原有功能)
# ==========================================

def define_barnsley_fern():
    """
    Defines the 4 affine transformations for the Barnsley Fern.
    Returns functions and their probabilities.
    """
    # Transformation 1: Stem (Probability 0.01)
    f1 = lambda x, y: (0.0, 0.16 * y)

    # Transformation 2: Successively smaller leaflets (Probability 0.85)
    f2 = lambda x, y: (0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6)

    # Transformation 3: Largest left-hand leaflet (Probability 0.07)
    f3 = lambda x, y: (0.2 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6)

    # Transformation 4: Largest right-hand leaflet (Probability 0.07)
    f4 = lambda x, y: (-0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44)

    funcs = [f1, f2, f3, f4]
    probs = [0.01, 0.85, 0.07, 0.07]
    return funcs, probs


def define_sierpinski_triangle():
    """
    Defines the 3 affine transformations for the Sierpinski Triangle.
    """
    # Three contractions towards the vertices of an equilateral triangle
    f1 = lambda x, y: (0.5 * x, 0.5 * y)  # Bottom left
    f2 = lambda x, y: (0.5 * x + 0.5, 0.5 * y)  # Bottom right
    f3 = lambda x, y: (0.5 * x + 0.25, 0.5 * y + 0.5)  # Top

    funcs = [f1, f2, f3]
    probs = [1/3, 1/3, 1/3]  # Equal probabilities
    return funcs, probs


def define_koch_curve():
    """
    Defines the 4 affine transformations for the Koch Curve.
    """
    # Hoist the rotation constants: computing them inside the lambdas costs
    # several microseconds per point.
    c = np.cos(np.pi / 3.0)
    s = np.sin(np.pi / 3.0)
    c3 = c / 3.0
    s3 = s / 3.0
    offset = np.sqrt(3.0) / 6.0

    # Four transformations that create the Koch curve pattern
    f1 = lambda x, y: (x/3, y/3)  # First segment
    f2 = lambda x, y: (x * c3 - y * s3 + 1/3,
                       x * s3 + y * c3)  # Second segment with rotation
    f3 = lambda x, y: (x * c3 + y * s3 + 1/2,
                       -x * s3 + y * c3 + offset)  # Third segment
    f4 = lambda x, y: (x/3 + 2/3, y/3)  # Fourth segment

    funcs = [f1, f2, f3, f4]
    probs = [0.25, 0.25, 0.25, 0.25]  # Equal probabilities
    return funcs, probs


def define_cantor_dust():
    """
    Defines the 4 affine transformations for Cantor Dust.
    A 2D generalization of the Cantor set.
    """
    # Four transformations that remove the central square
    scale = 1/3
    f1 = lambda x, y: (scale * x, scale * y)  # Bottom left
    f2 = lambda x, y: (scale * x + 2*scale, scale * y)  # Bottom right
    f3 = lambda x, y: (scale * x, scale * y + 2*scale)  # Top left
    f4 = lambda x, y: (scale * x + 2*scale, scale * y + 2*scale)  # Top right

    funcs = [f1, f2, f3, f4]
    probs = [0.25, 0.25, 0.25, 0.25]  # Equal probabilities
    return funcs, probs


def define_levy_curve():
    """
    Defines the 2 affine transformations for the Lévy C Curve.
    """
    # Two transformations that create the Lévy C curve pattern
    f1 = lambda x, y: (0.5 * x - 0.5 * y, 0.5 * x + 0.5 * y)  # Rotation and scaling
    f2 = lambda x, y: (0.5 * x + 0.5 * y + 0.5, -0.5 * x + 0.5 * y + 0.5)  # Rotation, scaling, translation

    funcs = [f1, f2]
    probs = [0.5, 0.5]  # Equal probabilities
    return funcs, probs


def define_dragon_curve():
    """
    Defines the 2 affine transformations for the Dragon Curve.
    """
    # Two transformations that create the dragon curve pattern
    f1 = lambda x, y: (0.5 * x - 0.5 * y, 0.5 * x + 0.5 * y)  # First transformation
    f2 = lambda x, y: (-0.5 * x - 0.5 * y + 1, 0.5 * x - 0.5 * y)  # Second transformation

    funcs = [f1, f2]
    probs = [0.5, 0.5]  # Equal probabilities
    return funcs, probs


def define_ifs_tree():
    """
    Defines a simple tree-like fractal using IFS.
    """
    # Four transformations creating a branching pattern
    f1 = lambda x, y: (0.05 * x, 0.6 * y)  # Trunk
    f2 = lambda x, y: (0.45 * x - 0.45 * y, 0.45 * x + 0.45 * y + 0.6)  # Left branch
    f3 = lambda x, y: (0.45 * x + 0.45 * y, -0.45 * x + 0.45 * y + 0.6)  # Right branch
    f4 = lambda x, y: (0.5 * x, 0.5 * y + 0.5)  # Central growth

    funcs = [f1, f2, f3, f4]
    probs = [0.1, 0.3, 0.3, 0.3]  # Different probabilities for natural look
    return funcs, probs


def define_spiral_ifs():
    """
    Defines a spiral pattern using IFS.
    """
    # Creates a spiral pattern with rotation and scaling.
    # The four trigonometric values are hoisted out of the lambdas and
    # combined with the scale factor.
    angle = np.pi / 6  # 30 degrees
    scale = 0.9
    a = scale * np.cos(angle)
    b = scale * np.sin(angle)
    f1 = lambda x, y: (a * x - b * y, b * x + a * y)
    f2 = lambda x, y: (a * x + b * y + 0.5, -b * x + a * y + 0.5)

    funcs = [f1, f2]
    probs = [0.5, 0.5]  # Equal probabilities
    return funcs, probs


class RandomFractalGenerator:
    """
    随机分形函数生成器。
    生成随机的压缩映射和概率分布来创建分形。
    """

    def __init__(self):
        self.n: int = None
        self.matrices: List[List[List[float]]] = None
        self.translations: List[List[float]] = None
        self.probabilities: List[float] = None
        self.functions: List[Callable[[float, float], Tuple[float, float]]] = None

    def generate_random_fractal(self) -> Dict[str, Any]:
        """
        生成一个随机的分形系统。

        Returns:
            Dict containing all generated parameters and functions
        """
        # 1. 随机选择n (2~5之间)
        self.n = np.random.randint(2, 6) # 6 is exclusive, so 2, 3, 4, 5
        print(f"生成 {self.n} 个压缩映射")

        # 2. 生成n个随机的压缩映射矩阵和对应的平移向量
        self.matrices = []
        self.translations = []
        self.functions = []

        for i in range(self.n):
            # 生成随机矩阵，确保是压缩映射（谱范数 < 1）
            matrix = self._generate_contraction_matrix()
            translation = np.random.uniform(-1, 1, 2) # 平移向量在[-1, 1]之间

            self.matrices.append(matrix.tolist())
            self.translations.append(translation.tolist())

            # 创建对应的函数 (仿射变换: x' = A*x + b)
            # 使用闭包捕获当前的matrix和translation值
            func = lambda x, y, m=matrix, t=translation: (
                m[0, 0] * x + m[0, 1] * y + t[0],
                m[1, 0] * x + m[1, 1] * y + t[1]
            )
            self.functions.append(func)

        # 3. 生成概率数组，倾向于等概率分布
        self.probabilities = self._generate_probabilities()

        return {
            'n': self.n,
            'matrices': self.matrices,
            'translations': self.translations,
            'probabilities': self.probabilities,
            'functions': self.functions # 注意：函数本身不能直接序列化到JSON
        }

    def _generate_contraction_matrix(self) -> np.ndarray:
        """
        生成一个压缩映射矩阵（谱范数 < 1）。
        """
        max_attempts = 100
        for attempt in range(max_attempts):
            # 生成随机矩阵，元素在[-1, 1]之间
            matrix = np.random.uniform(-1, 1, (2, 2))

            # 计算谱范数（最大奇异值）
            spectral_norm = np.linalg.norm(matrix, 2)

            # 归一化后谱范数恰好等于压缩因子，不需要再做一次 SVD。
            if spectral_norm > 0:
                contraction_factor = np.random.uniform(0.3, 0.8)
                return matrix / spectral_norm * contraction_factor

        # 如果多次尝试失败，返回一个已知的、安全的压缩映射
        print("Warning: Failed to generate a random contraction matrix after many attempts. Using a default one.")
        return np.array([[0.5, 0], [0, 0.5]])

    def _generate_probabilities(self) -> List[float]:
        """
        生成概率数组，倾向于等概率分布。
        """
        # 生成接近均匀分布的随机权重，范围在[0.8, 1.2]之间
        weights = np.random.uniform(0.8, 1.2, self.n)

        # 归一化，使总和为1
        probabilities = weights / np.sum(weights)

        return probabilities.tolist()

    def save_parameters(self, filename: str):
        """
        保存生成的参数到JSON文件。
        """
        if self.n is None:
            raise ValueError("请先生成随机分形 (call generate_random_fractal()).")

        # functions对象不能直接序列化，所以不包含在保存的参数中
        parameters = {
            'n_mappings': self.n,
            'matrices': self.matrices,
            'translations': self.translations,
            'probabilities': self.probabilities,
            'generation_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(parameters, f, indent=2, ensure_ascii=False)

        print(f"参数已保存到 {filename}")


def generate_matplotlib_color():
    """
    随机生成不浅不深、适用于matplotlib绘图的颜色。
    """
    r = np.random.uniform(0.2, 0.8)
    g = np.random.uniform(0.2, 0.8)
    b = np.random.uniform(0.2, 0.8)
    return (r, g, b)

def plot_all_functions():
    function_list = [
        define_cantor_dust, define_levy_curve, define_dragon_curve, define_ifs_tree,
        define_sierpinski_triangle, define_barnsley_fern, define_spiral_ifs, define_koch_curve
    ]

    for func in function_list:
        ifs_funcs, ifs_probs = func()

        # 创建可视化器实例
        visualizer = IFSVisualizer(ifs_funcs, ifs_probs)

        # 生成点集
        visualizer.generate_points(
            n_total=300000,
            m_discard=3000,
            initial_point=(0.1, 0.1)
        )

        visualizer.plot_matplotlib(
            save=True,
            filename=f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.png',
            color=generate_matplotlib_color(),
            show=False,
            close=True,
        )

def my_ifs():
    # Transformation 1: Stem
    f1 = lambda x, y: (0.0, 0.16 * y)

    # Transformation 2: Successively smaller leaflets
    # math.sin avoids the overhead of a NumPy ufunc call on a scalar.
    f2 = lambda x, y: (0.85 * x + 0.08*math.sin(y), 0.85 * y + 1.6)

    # Transformation 3: Largest left-hand leaflet
    f3 = lambda x, y: (0.2 * x - 0.31 * y, 0.25 * x + 0.2 * y + 1.6)

    # Transformation 4: Largest right-hand leaflet
    f4 = lambda x, y: (-0.15 * x + 0.33 * y, 0.28 * x + 0.22 * y + 0.44)

    funcs = [f1, f2, f3, f4]
    probs = [0.02, 0.84, 0.07, 0.07]
    return funcs, probs


if __name__ == "__main__":
    visualizer = IFSVisualizer(*my_ifs())
    visualizer.generate_points(1000000, 2000, initial_point=(0.1, 0.1))
    visualizer.plot_matplotlib(save=True, color="gold", alpha=0.2, filename="IFS.png")