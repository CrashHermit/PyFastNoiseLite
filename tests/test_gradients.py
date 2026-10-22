import numpy as np
import pytest

from pyfastnoiselite.pyfastnoiselite import (
    FastNoiseLite,
    FractalType,
    NoiseType,
    RotationType3D,
)


def _source_orientation(rotation):
    a, b = -0.211324865405187, 0.577350269189626
    if rotation == RotationType3D.RotationType3D_None:
        return np.full((3, 3), 2.0 / 3.0) - np.eye(3)
    if rotation == RotationType3D.RotationType3D_ImproveXYPlanes:
        return np.array([[1 + a, a, -b], [a, 1 + a, -b], [b, b, b]])
    return np.array([[1 + a, -b, a], [b, b, b], [a, -b, 1 + a]])


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


@pytest.mark.parametrize(
    "rotation",
    [
        RotationType3D.RotationType3D_None,
        RotationType3D.RotationType3D_ImproveXYPlanes,
        RotationType3D.RotationType3D_ImproveXZPlanes,
    ],
)
@pytest.mark.parametrize("frequency", [0.037, -0.037])
def test_gradient_is_with_respect_to_original_input_coordinates(rotation, frequency):
    noise = FastNoiseLite(seed=2026)
    noise.frequency = frequency
    noise.rotation_type_3d = rotation
    noise.noise_type = NoiseType.NoiseType_OpenSimplex2S
    noise.fractal_type = FractalType.FractalType_None
    point = np.array([7.2, -11.1, 20.05], dtype=np.float32)
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


@pytest.mark.parametrize(
    "rotation",
    [
        RotationType3D.RotationType3D_None,
        RotationType3D.RotationType3D_ImproveXYPlanes,
        RotationType3D.RotationType3D_ImproveXZPlanes,
    ],
)
def test_open_simplex2s_value_and_gradient_are_continuous_at_corner_ties(rotation):
    noise = FastNoiseLite(seed=-1337)
    noise.frequency = 0.37
    noise.rotation_type_3d = rotation
    noise.noise_type = NoiseType.NoiseType_OpenSimplex2S
    noise.fractal_type = FractalType.FractalType_None
    frequency = float(np.float32(0.37))
    default = _source_orientation(RotationType3D.RotationType3D_None)
    orientation = _source_orientation(rotation)
    source = frequency * (default @ [-1.02741, -1.75862226094, 0.533843])
    direction = frequency * default[:, 1]

    for permutation in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        q = source[list(permutation)]
        offset = 1e-4 * direction[list(permutation)]
        minus = np.asarray(orientation.T @ (q - offset) / frequency, dtype=np.float32)
        plus = np.asarray(orientation.T @ (q + offset) / frequency, dtype=np.float32)
        minus_value, minus_gradient = noise.get_open_simplex2s_with_gradient(*minus)
        plus_value, plus_gradient = noise.get_open_simplex2s_with_gradient(*plus)
        assert minus_value == noise.get_noise(*minus)
        assert plus_value == noise.get_noise(*plus)
        displacement = plus.astype(np.float64) - minus
        mean_gradient = (np.asarray(minus_gradient) + plus_gradient) * 0.5
        expected_change = mean_gradient @ displacement
        assert abs((plus_value - minus_value) - expected_change) < 2e-5
        np.testing.assert_allclose(plus_gradient, minus_gradient, rtol=0, atol=2e-2)


@pytest.mark.parametrize(
    "rotation",
    [
        RotationType3D.RotationType3D_None,
        RotationType3D.RotationType3D_ImproveXYPlanes,
        RotationType3D.RotationType3D_ImproveXZPlanes,
    ],
)
def test_support_and_cell_transitions_preserve_values_and_gradients(rotation):
    noise = FastNoiseLite(seed=2026)
    noise.frequency = 1.0
    noise.rotation_type_3d = rotation
    orientation = _source_orientation(rotation)
    for source in (
        [-1.0, 0.23, 0.37],
        [0.5, 0.23, 0.37],
        [0.5, 0.5, 0.5],
        [np.sqrt(0.75), 0.0, 0.0],
    ):
        points = np.asarray(
            [
                orientation.T @ (np.asarray(source) + sign * np.array([1e-6, 0, 0]))
                for sign in (-1, 1)
            ],
            dtype=np.float32,
        )
        values, gradients = noise.gen_open_simplex2s_with_gradient_from_coords(points.T)
        assert abs(float(values[1]) - values[0]) < 2e-5
        np.testing.assert_allclose(gradients[1], gradients[0], rtol=0, atol=2e-3)


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


@pytest.mark.parametrize(
    "rotation",
    [
        RotationType3D.RotationType3D_None,
        RotationType3D.RotationType3D_ImproveXYPlanes,
        RotationType3D.RotationType3D_ImproveXZPlanes,
    ],
)
@pytest.mark.parametrize("frequency", [0.037, -0.061])
def test_hessian_matches_gradient_differences_and_batch(rotation, frequency):
    noise = FastNoiseLite(seed=907)
    noise.frequency = frequency
    noise.rotation_type_3d = rotation
    noise.noise_type = NoiseType.NoiseType_Perlin
    noise.fractal_type = FractalType.FractalType_FBm
    points = np.asarray([[0.37, -1.21], [-0.81, 0.29], [1.13, 2.7]], dtype=np.float32)
    values, gradients, hessians = noise.gen_open_simplex2s_with_hessian_from_coords(points)
    assert values.shape == (2,)
    assert gradients.shape == (2, 3)
    assert hessians.shape == (2, 3, 3)
    assert values.dtype == gradients.dtype == hessians.dtype == np.float32
    step = np.float32(0.001)
    for index in range(points.shape[1]):
        value, gradient, hessian = noise.get_open_simplex2s_with_hessian(*points[:, index])
        raw_value, raw_gradient = noise.get_open_simplex2s_with_gradient(*points[:, index])
        assert values[index] == value == raw_value
        np.testing.assert_array_equal(gradients[index], raw_gradient)
        np.testing.assert_array_equal(gradients[index], gradient)
        np.testing.assert_array_equal(hessians[index], hessian)
        np.testing.assert_allclose(hessian, np.asarray(hessian).T, rtol=0, atol=2e-6)
        finite_difference = np.empty((3, 3), dtype=np.float64)
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float32)
            offset[axis] = step
            plus_gradient = noise.get_open_simplex2s_with_gradient(*(points[:, index] + offset))[1]
            minus_gradient = noise.get_open_simplex2s_with_gradient(*(points[:, index] - offset))[1]
            finite_difference[:, axis] = (
                np.asarray(plus_gradient) - np.asarray(minus_gradient)
            ) / (2.0 * float(step))
        np.testing.assert_allclose(hessian, finite_difference, rtol=3e-2, atol=5e-3)


def test_hessian_support_join_and_empty_batch():
    noise = FastNoiseLite(seed=2026)
    noise.frequency = 1.0
    orientation = _source_orientation(RotationType3D.RotationType3D_None)
    points = np.asarray(
        [
            orientation.T @ [np.sqrt(0.75) - 1e-5, 0.0, 0.0],
            orientation.T @ [np.sqrt(0.75) + 1e-5, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).T
    _, gradients, hessians = noise.gen_open_simplex2s_with_hessian_from_coords(points)
    np.testing.assert_allclose(gradients[1], gradients[0], rtol=0, atol=2e-3)
    np.testing.assert_allclose(hessians[1], hessians[0], rtol=0, atol=2e-2)
    values, gradients, hessians = noise.gen_open_simplex2s_with_hessian_from_coords(
        np.empty((3, 0), dtype=np.float32)
    )
    assert values.shape == (0,)
    assert gradients.shape == (0, 3)
    assert hessians.shape == (0, 3, 3)
