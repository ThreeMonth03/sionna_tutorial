"""Physical and model constants used throughout the tutorial."""

from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
TWO_PI = 2.0 * np.pi

# 3GPP TR 38.901 Table 7.5-3: normalized ray offsets within a cluster.
RAY_OFFSETS = np.array(
    [
        0.0447,
        -0.0447,
        0.1413,
        -0.1413,
        0.2492,
        -0.2492,
        0.3715,
        -0.3715,
        0.5129,
        -0.5129,
        0.6797,
        -0.6797,
        0.8844,
        -0.8844,
        1.1481,
        -1.1481,
        1.5195,
        -1.5195,
        2.1551,
        -2.1551,
    ],
    dtype=np.float64,
)

RAYS_PER_CLUSTER = int(RAY_OFFSETS.size)
