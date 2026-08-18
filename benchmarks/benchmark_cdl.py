"""Measure end-to-end CDL generation on NumPy or CuPy.

The timed region includes coefficient construction but reuses the same sampled
random state so CPU/GPU results represent the same physical realization.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from sionna_tutorial import AntennaArray, CDLChannel, get_backend


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cdl_benchmark.csv"))
    args = parser.parse_args()

    backend = get_backend(args.backend)
    carrier = 3.5e9
    channel = CDLChannel(
        "C",
        100e-9,
        carrier,
        tx_array=AntennaArray.ula(4, carrier),
        rx_array=AntennaArray.ula(4, carrier),
    )

    rows: list[dict[str, object]] = []
    for batch in args.batches:
        state = channel.sample_state(batch, seed=123)
        kwargs = dict(
            batch_size=batch,
            num_time_steps=args.time_steps,
            sampling_frequency_hz=2_000.0,
            velocity_mps=(10.0, 0.0, 0.0),
            random_state=state,
            backend=backend,
        )
        channel.generate(**kwargs)
        backend.synchronize()

        samples: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = channel.generate(**kwargs)
            backend.synchronize()
            samples.append(time.perf_counter() - start)
            del result

        median = percentile(samples, 50.0)
        row = {
            "backend": backend.name,
            "batch": batch,
            "time_steps": args.time_steps,
            "median_ms": median * 1e3,
            "p10_ms": percentile(samples, 10.0) * 1e3,
            "p90_ms": percentile(samples, 90.0) * 1e3,
            "channels_per_second": batch / median,
        }
        rows.append(row)
        print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
