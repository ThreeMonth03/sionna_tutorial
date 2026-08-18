# 3GPP MIMO Channel Tutorial

A readable, test-driven NumPy/CuPy prototype for learning how 3GPP TR 38.901
TDL/CDL channels turn path delays, angles, antenna geometry, polarization, and
mobility into a time-varying MIMO channel.

This repository is designed to be **read before it is optimized**. The NumPy
and CuPy paths execute the same equations and can reuse the exact same sampled
random state, making CPU/GPU differential testing straightforward.

> **Status:** educational prototype, not an official 3GPP conformance tool and
> not affiliated with NVIDIA. The code implements the small-scale channel core;
> it deliberately does not implement a complete UMi/UMa/RMa scenario engine or
> a complete 5G NR modem.

## What is implemented

- TDL-A/B/C/D/E standardized delay and power profiles
- CDL-A/B/C/D/E standardized delay, power, angle, spread, XPR, and LOS profiles
- Twenty rays per CDL cluster using TR 38.901 Table 7.5-3 offsets
- Independent random coupling of AoA, AoD, ZoA, and ZoD inside each cluster
- ULA and UPA antenna-port geometry, including dual-polarized ports
- Omni and normalized TR 38.901-style element patterns
- 2×2 polarization/Jones matrix with cross-polarization ratio
- Array-position phase, receiver/transmitter mobility, and Doppler evolution
- LOS/Rician combination for CDL-D/E and TDL-D/E
- Sparse channel impulse response (CIR) to OFDM frequency response (CFR)
- A minimal QPSK + MIMO + perfect-CSI zero-forcing OFDM visualization
- NumPy CPU and optional CuPy CUDA execution
- Unit tests for profiles, rays, geometry, channel power, Doppler, and OFDM

## Deliberate simplifications

The first version leaves these pieces visible as future extensions rather than
hiding them behind a false claim of full conformance:

- no UMi, UMa, RMa, indoor, path-loss, shadow-fading, or LOS-probability logic
- no global/local coordinate-system rotations or arbitrary device orientation
- no spatial consistency while a terminal moves through a geographical scene
- no system-level subclustering of the two strongest clusters
- no near-field, spatial non-stationarity, or Release-19 FR3 extensions
- TDL time variation uses a compact Jakes-style sum-of-sinusoids for teaching
- no PUSCH, LDPC, DMRS, channel estimator, HARQ, scheduler, or protocol stack
- no custom CUDA kernels, SOLVCON integration, multi-GPU scheduling, or tuning

Those boundaries are intentional: the repository focuses on the numerical
meaning of a clustered MIMO channel.

## Installation from a clean Python environment

The tested Python versions are 3.11, 3.12, and 3.13. The repository provides a
pinned `requirements.txt` containing the project itself, NumPy, Matplotlib,
CuPy for CUDA 12, and all test/lint tools. No preinstalled Python packages are
assumed.

```bash
git clone https://github.com/ThreeMonth03/sionna_tutorial.git
cd sionna_tutorial

python3 -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

`requirements.txt` installs `cupy-cuda12x[ctk]`. The `[ctk]` extra installs the
CUDA 12 runtime/component wheels into the Python environment, so only a
compatible NVIDIA driver is required; a separately installed system CUDA
Toolkit is not required. Do not install another `cupy` or `cupy-cuda*` package
in the same environment.

Verify that Python sees the GPU:

```bash
python3 -c "import cupy as cp; print('GPU count:', cp.cuda.runtime.getDeviceCount()); print('GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'])"
```

Run the complete test suite, including the CUDA differential test when a GPU is
available:

```bash
python3 -m pytest -q
```

The RTX 2060 (Turing, compute capability 7.5) is sufficient for all examples.
The examples default to `--backend auto`, so they fall back to NumPy when CuPy
is unavailable.

## Run the examples

Read and run them in numerical order:

```bash
python3 examples/01_single_path.py
python3 examples/02_multipath_mimo.py --backend numpy
python3 examples/03_tdl_profiles.py
python3 examples/04_cdl_channel.py --model C --backend auto
python3 examples/05_mobility_doppler.py --backend auto
python3 examples/06_ofdm_demo.py --backend auto --snr-db 20
```

Figures are written to `artifacts/`.

| Example | Main question |
|---|---|
| `01_single_path.py` | Why do antenna position and motion rotate complex phase? |
| `02_multipath_mimo.py` | How do many rays create one MIMO matrix? |
| `03_tdl_profiles.py` | What do standardized delay/power profiles look like? |
| `04_cdl_channel.py` | How do clusters, rays, polarization, and arrays form a CIR? |
| `05_mobility_doppler.py` | Why does a fast terminal make the channel vary faster? |
| `06_ofdm_demo.py` | How does the CIR distort QPSK subcarriers, and how does ZF undo it? |

## Minimal API

```python
from sionna_tutorial import AntennaArray, CDLChannel

fc = 3.5e9
channel = CDLChannel(
    model="C",
    delay_spread_s=100e-9,
    carrier_frequency_hz=fc,
    tx_array=AntennaArray.ula(4, fc),
    rx_array=AntennaArray.ula(4, fc),
)

result = channel.generate(
    batch_size=128,
    num_time_steps=32,
    sampling_frequency_hz=2_000.0,
    velocity_mps=(30 / 3.6, 0.0, 0.0),
    seed=42,
    backend="numpy",  # or "cupy" / "auto"
)

print(result.coefficients.shape)
# [batch, rx_antenna, tx_antenna, path, time]
print(result.delays_s.shape)
# [path]
```

For an exact CPU/GPU comparison, sample once and reuse the state:

```python
state = channel.sample_state(batch_size=128, seed=42)
cpu = channel.generate(128, random_state=state, backend="numpy")
gpu = channel.generate(128, random_state=state, backend="cupy")
```

## Tensor shape map

The most important tensors are deliberately documented in code:

```text
cluster centers                       [cluster]
ray angles after offsets/coupling     [batch, cluster, ray]
polarization phases                   [batch, cluster, ray, 4]
per-path channel coefficient          [batch, rx, tx, path, time]
OFDM frequency response               [batch, rx, tx, time, subcarrier]
```

The deterministic CDL coefficient engine processes one cluster at a time. It
avoids materializing the full
`batch × cluster × ray × rx × tx × time` tensor while keeping the equations
readable.

## Validate and benchmark

```bash
python3 -m ruff check .
python3 -m pytest -q

python3 benchmarks/benchmark_cdl.py \
  --backend numpy \
  --batches 1 8 64 256 \
  --time-steps 8

python3 benchmarks/benchmark_cdl.py \
  --backend cupy \
  --batches 1 8 64 256 \
  --time-steps 8
```

The benchmark includes the deterministic coefficient construction and reuses a
fixed random state. It synchronizes CUDA before stopping the timer.

## Recommended reading path

1. [`docs/algorithm.md`](docs/algorithm.md)
2. `src/sionna_tutorial/geometry.py`
3. `src/sionna_tutorial/rays.py`
4. `src/sionna_tutorial/coefficients.py`
5. `src/sionna_tutorial/cdl.py`
6. `src/sionna_tutorial/ofdm.py`
7. the corresponding unit tests

## References and attribution

- 3GPP TR 38.901, channel models for 0.5–100 GHz
- NVIDIA Sionna, an Apache-2.0 open-source communication-system research library

The packaged profile tables are derived from Sionna's Apache-2.0 model files;
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
