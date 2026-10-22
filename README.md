[![Build Status](https://travis-ci.org/tizilogic/PyFastNoiseLite.svg?branch=master)](https://travis-ci.org/tizilogic/PyFastNoiseLite)

# Cython wrapper for the [FastNoise Lite](https://github.com/Auburn/FastNoise/) library

This wraps Auburns' great [FastNoise Lite](https://github.com/Auburn/FastNoise/)
library using Cython. This fork targets Python 3.13+.

## Installation

Install this fork, including its pinned native submodule:

```bash
pip install "pyfastnoiselite @ git+https://github.com/CrashHermit/PyFastNoiseLite.git@analytic-opensimplex2s-gradients"
```

Source builds require a C++11 compiler; isolated builds install Cython, NumPy,
setuptools, and wheel. Native headers come from the pinned submodule rather than
a build-time download of an upstream branch.

> __Note__:
>
> This wrapper currently lacks the domain warping functionality and
> documentation. Both are planned to be added in the future.

## Usage

```py
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType

# Initializing with seed
noise = FastNoiseLite(seed=1337)
# Set noise type (optional, defaults to OpenSimplex2)
noise.noise_type = NoiseType.NoiseType_OpenSimplex2S

# Get 2D noise
print(noise.get_noise(34, 22)) # 0.7130074501037598
print(noise.get_noise(100, 110)) # -0.614801287651062

# Get 3D noise
print(noise.get_noise(95, 100, 30)) # -0.45421651005744934


import numpy as np

Xs = [3, 57, 95]
Ys = [4, 13, 100]
Zs = [0, -4, 30]
coords = np.array([Xs, Ys, Zs], dtype=np.float32)
# Generate noise for each coordinate
print(noise.gen_from_coords(coords)) # [0.11303097 -0.5841235  -0.45224023]
```

## OpenSimplex2S gradients

`FastNoiseLite` also exposes `get_open_simplex2s_with_gradient(x, y, z)`,
which returns `(value, (dx, dy, dz))`, and
`gen_open_simplex2s_with_gradient_from_coords(coords)`, which returns
`(values, gradients)`. The batch input is a float32 array with shape `(3, N)`
(one XYZ coordinate per column); outputs have shapes `(N,)` and `(N, 3)`,
respectively, and use float32 CPU arithmetic. Empty batches are supported.

These APIs explicitly evaluate raw 3D OpenSimplex2S with the instance's seed,
frequency, and 3D rotation setting. The value is the same as `get_noise` when
the instance uses OpenSimplex2S with `FractalType_None`. Derivatives are
Cartesian derivatives with respect to the original input coordinates, including
the frequency and rotation transformations. Other noise types and fractal
compositions are not evaluated, and no full fractal-gradient API is provided.

The 3D OpenSimplex2S value-only sampler and these gradient APIs use complete
active-corner support from the two offset cubic lattices. This corrects
corner-selection discontinuities in the previous 3D implementation. The hash,
gradient table, and normalization are retained, but corrected noise values
intentionally differ where the old selection omitted contributing corners.
The underlying compact-support polynomial contributions join continuously in
real arithmetic; float precision still limits the numerical behavior observed
at runtime. This change concerns 3D OpenSimplex2S only: it does not change 2D
OpenSimplex2S or other noise algorithms.
