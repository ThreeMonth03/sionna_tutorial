"""Small backend adapter shared by the NumPy and CuPy implementations.

The prototype intentionally keeps the adapter tiny. The communication
algorithm remains visible instead of being hidden behind a compiler/runtime
abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

BackendName = Literal["numpy", "cupy", "auto"]


@dataclass(frozen=True)
class Backend:
    """Container for a NumPy-compatible array namespace."""

    name: str
    xp: Any

    @property
    def is_gpu(self) -> bool:
        return self.name == "cupy"

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        return self.xp.asarray(value, dtype=dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self.name == "cupy":
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def synchronize(self) -> None:
        if self.name == "cupy":
            self.xp.cuda.get_current_stream().synchronize()


def cupy_available() -> bool:
    """Return whether CuPy can be imported and a CUDA device is usable."""

    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_backend(name: BackendName = "auto") -> Backend:
    """Select NumPy or CuPy.

    ``auto`` uses CuPy only when importing it and querying the CUDA runtime both
    succeed. This makes examples portable to CPU-only CI runners.
    """

    if name == "numpy":
        return Backend("numpy", np)

    if name in {"cupy", "auto"}:
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                return Backend("cupy", cp)
        except Exception:
            if name == "cupy":
                raise RuntimeError(
                    "CuPy/CUDA was requested but no working CUDA installation was found. "
                    "Install the matching CuPy wheel, for example `pip install cupy-cuda12x`."
                ) from None

    return Backend("numpy", np)
