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

## Analytic OpenSimplex2S derivatives

`FastNoiseLite` exposes raw 3D OpenSimplex2S sampling with first and second
derivatives. `get_open_simplex2s_with_gradient(x, y, z)` returns
`(value, (dx, dy, dz))`; `get_open_simplex2s_with_hessian(x, y, z)` returns
`(value, (dx, dy, dz), ((dxx, dxy, dxz), (dyx, dyy, dyz), (dzx, dzy, dzz)))`.
The batch counterparts return `(values, gradients)` and
`(values, gradients, Hessians)` respectively. Batch input is a float32 array
with shape `(3, N)` (one XYZ coordinate per column); output shapes are `(N,)`,
`(N, 3)`, and `(N, 3, 3)`. Empty batches are supported.

The APIs evaluate raw 3D OpenSimplex2S using the instance seed, frequency, and
3D rotation, irrespective of selected noise/fractal modes. Derivatives are
Cartesian derivatives with respect to original input coordinates, including
both axes of the frequency/orientation transform. Values and gradients match
the corresponding existing APIs exactly. The Hessian is analytic and symmetric
up to floating-point rounding; finite differences are used only in tests.

The 3D OpenSimplex2S value-only sampler and these gradient APIs use complete
active-corner support from the two offset cubic lattices. This corrects
corner-selection discontinuities in the previous 3D implementation. The hash,
gradient table, and normalization are retained, but corrected noise values
intentionally differ where the old selection omitted contributing corners.
The underlying compact-support polynomial contributions join continuously in
real arithmetic; float precision still limits the numerical behavior observed
at runtime. This change concerns 3D OpenSimplex2S only: it does not change 2D
OpenSimplex2S or other noise algorithms.
