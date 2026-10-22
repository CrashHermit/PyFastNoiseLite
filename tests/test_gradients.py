import numpy as np
import pytest

from pyfastnoiselite.pyfastnoiselite import (
    FastNoiseLite,
    FractalType,
    NoiseType,
    RotationType3D,
)


@pytest.mark.parametrize(
    "rotation",
    [
        RotationType3D.RotationType3D_None,
        RotationType3D.RotationType3D_ImproveXYPlanes,
        RotationType3D.RotationType3D_ImproveXZPlanes,
    ],
)
@pytest.mark.parametrize("frequency", [0.013, 0.071])
def test_raw_gradient_matches_scalar_and_batch(rotation, frequency):
    noise = FastNoiseLite(seed=417)
    noise.frequency = frequency
    noise.rotation_type_3d = rotation
    noise.noise_type = NoiseType.NoiseType_OpenSimplex2S
    noise.fractal_type = FractalType.FractalType_None
    points = np.array(
        [[0.37, -1.21, 3.4, 0.0], [-0.81, 0.29, 1.5, -2.3], [1.13, 2.7, -0.46, 0.91]],
        dtype=np.float32,
    )

    values, gradients = noise.gen_open_simplex2s_with_gradient_from_coords(points)
    assert values.shape == (points.shape[1],)
    assert gradients.shape == (points.shape[1], 3)
    assert values.dtype == gradients.dtype == np.float32
    for index in range(points.shape[1]):
        value, gradient = noise.get_open_simplex2s_with_gradient(*points[:, index])
        assert values[index] == value
        np.testing.assert_array_equal(gradients[index], gradient)
        assert value == noise.get_noise(*points[:, index])


def test_gradient_is_with_respect_to_original_input_coordinates():
    noise = FastNoiseLite(seed=2026)
    noise.frequency = 0.037
    noise.rotation_type_3d = RotationType3D.RotationType3D_ImproveXZPlanes
    noise.noise_type = NoiseType.NoiseType_OpenSimplex2S
    noise.fractal_type = FractalType.FractalType_None
    point = np.array([0.72, -1.1, 2.05], dtype=np.float32)
    value, gradient = noise.get_open_simplex2s_with_gradient(*point)
    step = np.float32(0.002)
    finite_difference = np.empty(3, dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float32)
        offset[axis] = step
        plus = noise.get_noise(*(point + offset))
        minus = noise.get_noise(*(point - offset))
        finite_difference[axis] = (plus - minus) / (2.0 * float(step))
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-2, atol=2e-3)


def test_raw_gradient_ignores_noise_type_and_fractal_selection():
    noise = FastNoiseLite(seed=51)
    noise.frequency = 0.021
    noise.rotation_type_3d = RotationType3D.RotationType3D_ImproveXYPlanes
    point = (1.4, -0.8, 2.2)
    raw = noise.get_open_simplex2s_with_gradient(*point)

    noise.noise_type = NoiseType.NoiseType_Perlin
    noise.fractal_type = FractalType.FractalType_FBm
    assert noise.get_open_simplex2s_with_gradient(*point) == raw


def test_gradient_batch_requires_xyz_rows_and_supports_empty_input():
    noise = FastNoiseLite()
    values, gradients = noise.gen_open_simplex2s_with_gradient_from_coords(
        np.empty((3, 0), dtype=np.float32)
    )
    assert values.shape == (0,)
    assert gradients.shape == (0, 3)
    with pytest.raises(ValueError, match=r"shape \(3, N\)"):
        noise.gen_open_simplex2s_with_gradient_from_coords(
            np.empty((2, 3), dtype=np.float32)
        )
