"""Tests for the IFS visualizer and its accelerated backends."""

import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import IFS


class TestAffineDetection(unittest.TestCase):
    """Check the automatic affine detection used by the fast backends."""

    def test_predefined_systems_are_affine(self):
        factories = [
            IFS.define_barnsley_fern,
            IFS.define_sierpinski_triangle,
            IFS.define_koch_curve,
            IFS.define_cantor_dust,
            IFS.define_levy_curve,
            IFS.define_dragon_curve,
            IFS.define_ifs_tree,
            IFS.define_spiral_ifs,
        ]
        for factory in factories:
            with self.subTest(factory=factory.__name__):
                visualizer = IFS.IFSVisualizer(*factory())
                self.assertIsNotNone(visualizer._ensure_affine())

    def test_my_ifs_is_non_affine(self):
        visualizer = IFS.IFSVisualizer(*IFS.my_ifs())
        self.assertIsNone(visualizer._ensure_affine())

    def test_random_generator_is_affine(self):
        generator = IFS.RandomFractalGenerator()
        params = generator.generate_random_fractal()
        visualizer = IFS.IFSVisualizer(
            params["functions"], params["probabilities"]
        )
        self.assertIsNotNone(visualizer._ensure_affine())


class TestValidation(unittest.TestCase):
    """Check argument validation."""

    def test_probability_length_mismatch(self):
        with self.assertRaises(ValueError):
            IFS.IFSVisualizer([lambda x, y: (x, y)], [0.5, 0.5])

    def test_negative_probability(self):
        with self.assertRaises(ValueError):
            IFS.IFSVisualizer([lambda x, y: (x, y)] * 2, [1.2, -0.2])

    def test_probability_sum(self):
        with self.assertRaises(ValueError):
            IFS.IFSVisualizer([lambda x, y: (x, y)] * 2, [0.2, 0.2])

    def test_empty_functions(self):
        with self.assertRaises(ValueError):
            IFS.IFSVisualizer([])

    def test_discard_not_larger_than_total(self):
        visualizer = IFS.IFSVisualizer(*IFS.define_dragon_curve())
        with self.assertRaises(ValueError):
            visualizer.generate_points(100, 100)

    def test_numpy_backend_rejects_non_affine(self):
        visualizer = IFS.IFSVisualizer(*IFS.my_ifs())
        with self.assertRaises(RuntimeError):
            visualizer.generate_points(1000, 10, method="numpy")


class TestGeneration(unittest.TestCase):
    """Check point counts, finiteness and agreement between backends."""

    def test_all_backends_return_expected_shape(self):
        funcs, probs = IFS.define_barnsley_fern()
        methods = ["python", "numpy"]
        if IFS.NUMBA_AVAILABLE:
            methods.append("numba")

        for method in methods:
            with self.subTest(method=method):
                visualizer = IFS.IFSVisualizer(funcs, probs)
                points = visualizer.generate_points(
                    20_000, 200, (0.1, 0.1), method=method, seed=7
                )
                self.assertEqual(points.shape, (19_800, 2))
                self.assertTrue(np.isfinite(points).all())
                self.assertEqual(visualizer.backend, method)

    def test_backends_agree_on_distribution(self):
        funcs, probs = IFS.define_barnsley_fern()
        reference = IFS.IFSVisualizer(funcs, probs).generate_points(
            150_000, 500, (0.1, 0.1), method="python", seed=11
        )
        candidate = IFS.IFSVisualizer(funcs, probs).generate_points(
            150_000, 500, (0.1, 0.1), method="numpy", seed=11
        )
        for axis in range(2):
            self.assertAlmostEqual(
                reference[:, axis].mean(), candidate[:, axis].mean(), delta=0.15
            )
            self.assertAlmostEqual(
                reference[:, axis].std(), candidate[:, axis].std(), delta=0.15
            )

    @unittest.skipUnless(IFS.NUMBA_AVAILABLE, "Numba is not installed")
    def test_non_affine_numba_backend(self):
        visualizer = IFS.IFSVisualizer(*IFS.my_ifs())
        points = visualizer.generate_points(
            30_000, 300, (0.1, 0.1), method="numba", seed=3
        )
        self.assertEqual(points.shape, (29_700, 2))
        self.assertEqual(visualizer.backend, "numba")
        self.assertTrue(np.isfinite(points).all())
        self.assertLess(points[:, 1].max(), 11.0)
        self.assertGreater(points[:, 1].min(), -1.0)

    def test_seed_is_reproducible(self):
        visualizer = IFS.IFSVisualizer(*IFS.define_dragon_curve())
        first = visualizer.generate_points(5000, 100, method="python", seed=123)
        second = visualizer.generate_points(5000, 100, method="python", seed=123)
        np.testing.assert_array_equal(first, second)


class TestRendering(unittest.TestCase):
    """Check the plotting helpers without opening a window."""

    def setUp(self):
        self.visualizer = IFS.IFSVisualizer(*IFS.define_dragon_curve())
        self.visualizer.generate_points(10_000, 100, seed=5)

    def test_density_plot_writes_a_small_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "density.png")
            self.visualizer.plot_matplotlib(
                save=True,
                filename=path,
                mode="density",
                bins=400,
                dpi=300,
                show=False,
                close=True,
            )
            self.assertTrue(os.path.exists(path))
            image = mpimg.imread(path)
            self.assertGreater(image.shape[0], 0)
            self.assertLessEqual(max(image.shape[:2]), 400)

    def test_plotly_downsampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fractal.html")
            figure = self.visualizer.plot_plotly(
                save=True,
                filename=path,
                max_points=1000,
                include_plotlyjs="cdn",
                show=False,
            )
            self.assertLessEqual(len(figure.data[0].x), 1001)
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
