from enum import Enum

import numpy as np
import numpy.typing as npt

type Gradient3D = tuple[float, float, float]
type Hessian3D = tuple[Gradient3D, Gradient3D, Gradient3D]

class NoiseType(Enum):
    NoiseType_OpenSimplex2: int
    NoiseType_OpenSimplex2S: int
    NoiseType_Cellular: int
    NoiseType_Perlin: int
    NoiseType_ValueCubic: int
    NoiseType_Value: int

class RotationType3D(Enum):
    RotationType3D_None: int
    RotationType3D_ImproveXYPlanes: int
    RotationType3D_ImproveXZPlanes: int

class FractalType(Enum):
    FractalType_None: int
    FractalType_FBm: int
    FractalType_Ridged: int
    FractalType_PingPong: int
    FractalType_DomainWarpProgressive: int
    FractalType_DomainWarpIndependent: int

class CellularDistanceFunction(Enum):
    CellularDistanceFunction_Euclidean: int
    CellularDistanceFunction_EuclideanSq: int
    CellularDistanceFunction_Manhattan: int
    CellularDistanceFunction_Hybrid: int

class CellularReturnType(Enum):
    CellularReturnType_CellValue: int
    CellularReturnType_Distance: int
    CellularReturnType_Distance2: int
    CellularReturnType_Distance2Add: int
    CellularReturnType_Distance2Sub: int
    CellularReturnType_Distance2Mul: int
    CellularReturnType_Distance2Div: int

class DomainWarpType(Enum):
    DomainWarpType_OpenSimplex2: int
    DomainWarpType_OpenSimplex2Reduced: int
    DomainWarpType_BasicGrid: int

class FastNoiseLite:
    def __init__(self, seed: int = 1337) -> None: ...
    seed: int
    frequency: float
    noise_type: NoiseType
    rotation_type_3d: RotationType3D
    fractal_type: FractalType
    fractal_octaves: int
    fractal_lacunarity: float
    fractal_gain: float
    fractal_weighted_strength: float
    fractal_ping_pong_strength: float
    cellular_distance_function: CellularDistanceFunction
    cellular_return_type: CellularReturnType
    cellular_jitter: float
    domain_warp_type: DomainWarpType
    domain_warp_amp: float

    def get_noise(self, x: float, y: float, z: float | None = ...) -> float: ...
    def gen_from_coords(
        self, coords: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]: ...
    def get_open_simplex2s_with_gradient(
        self, x: float, y: float, z: float
    ) -> tuple[float, Gradient3D]: ...
    def gen_open_simplex2s_with_gradient_from_coords(
        self, coords: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
    def get_open_simplex2s_with_hessian(
        self, x: float, y: float, z: float
    ) -> tuple[float, Gradient3D, Hessian3D]: ...
    def gen_open_simplex2s_with_hessian_from_coords(
        self, coords: npt.NDArray[np.float32]
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]: ...
