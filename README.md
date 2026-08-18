# 3GPP MIMO Channel Tutorial

A readable, test-driven NumPy/CuPy prototype for learning how a wireless
propagation model turns path delays, angles, antenna geometry, polarization,
and mobility into a time-varying MIMO channel.

## Start here: what part of communication does this repository implement?

This is **not** a complete communication-system tutorial, 5G protocol stack, or
5G NR modem. Its core is the block between the transmit and receive antennas:

```text
Application / TCP-IP / PDCP-RLC-MAC
                |
                v
PHY transmitter
coding -> QAM -> precoding -> OFDM
                |
                v
RF -> Tx antenna
                |
                v
====================================================
       WIRELESS PROPAGATION CHANNEL

   TDL / CDL / MIMO / multipath / Doppler

          THIS REPOSITORY'S CORE
====================================================
                |
                v
Rx antenna -> RF
                |
                v
PHY receiver
FFT -> channel estimation -> detection -> decoding
                |
                v
PDCP-RLC-MAC / IP / application
```

The repository mainly **generates** the channel

```text
h [batch, rx_antenna, tx_antenna, delayed_path, time]
```

and only provides one deliberately tiny example of applying that channel to
uncoded QPSK subcarriers.

Read these documents before reading the implementation:

1. **[`docs/00_system_overview.md`](docs/00_system_overview.md)** — complete
   communication-system map, repository coverage, and the distinction between
   channel generation, channel application, and channel estimation.
2. **[`docs/01_learning_path.md`](docs/01_learning_path.md)** — what Examples
   01-06 teach, their inputs/outputs, and what each one intentionally omits.
3. **[`docs/02_source_map.md`](docs/02_source_map.md)** — file roles, call graph,
   and tensor flow through the source tree.
4. **[`docs/algorithm.md`](docs/algorithm.md)** — formula-to-code explanation
   inside the channel generator.

## What is implemented

### Wireless-channel modeling

- TDL-A/B/C/D/E standardized delay and power profiles
- CDL-A/B/C/D/E standardized delay, power, angle, spread, XPR, and LOS profiles
- twenty rays per CDL cluster using TR 38.901 Table 7.5-3 offsets
- independent random coupling of AoA, AoD, ZoA, and ZoD inside each cluster
- ULA and UPA antenna-port geometry, including dual-polarized ports
- omni and normalized TR 38.901-style element patterns
- 2x2 polarization/Jones matrix with cross-polarization ratio
- array-position phase, receiver/transmitter mobility, and Doppler evolution
- LOS/Rician combination for CDL-D/E and TDL-D/E
- sparse channel impulse response (CIR) to OFDM frequency response (CFR)
- NumPy CPU and optional CuPy CUDA execution

### Tiny application around the channel

- uncoded QPSK symbols on frequency-domain OFDM subcarriers
- $\mathbf{Y}[k] = \mathbf{H}[k]\mathbf{X}[k] + \mathbf{N}[k]$
- perfect-CSI pseudo-inverse / zero-forcing equalization
- hard decisions and a small BER result

This application exists to make the channel tensor physically meaningful. It
is not a complete NR receiver.

## Current coverage inside a full link

| Block | Status |
|---|---|
| Protocol layers and MAC scheduling | Not implemented |
| CRC, LDPC, rate matching, HARQ | Not implemented |
| QAM | Minimal uncoded QPSK in Example 06 |
| Precoding / beamforming | Not implemented |
| Full OFDM waveform, IFFT, cyclic prefix | Not implemented |
| RF transmitter/receiver impairments | Not implemented |
| TDL/CDL propagation channel | Implemented |
| Antenna arrays, polarization, Doppler | Implemented |
| Pilot / DMRS | Not implemented |
| Channel estimation | Not implemented |
| MIMO equalization | Minimal perfect-CSI pseudo-inverse |
| Soft demapping and decoding | Not implemented |
| BLER Monte Carlo campaign | Not implemented |

## Three operations that are easy to confuse

### Channel generation

Generate the propagation model itself:

```python
realization = channel.generate(...)
```

Output:

```text
coefficients  [batch, rx, tx, path, time]
delays        [path]
```

Examples 01-05 focus on this.

### Channel application

Apply a known channel to transmitted symbols:

```math
\mathbf{y}[k]=\mathbf{H}[k]\mathbf{x}[k]+\mathbf{n}[k].
```

Example 06 does this in the frequency domain.

### Channel estimation

A real receiver does not know $\mathbf{H}$. It observes pilots and estimates
$\widehat{\mathbf{H}}$. This repository does not yet implement that step.
Example 06 directly gives the exact simulated $\mathbf{H}$ to the equalizer;
that idealization is called **perfect CSI**.

## Learning path: Examples 01-06

The examples are small teaching experiments, not performance stress tests.

| Example | System question | Main output |
|---|---|---|
| `01_single_path.py` | How do antenna position and motion rotate one ray's complex phase? | phase versus time |
| `02_multipath_mimo.py` | How do many rays create the matrix $\mathbf{H}$? | MIMO magnitude/phase heatmap |
| `03_tdl_profiles.py` | What delayed echoes are defined by TDL-A-E? | power-delay profiles |
| `04_cdl_channel.py` | How are clusters, rays, polarization, and arrays combined? | one CDL CIR realization |
| `05_mobility_doppler.py` | Why does a faster terminal make $\mathbf{H}(t)$ vary faster? | fading traces at three speeds |
| `06_ofdm_demo.py` | How does $\mathbf{H}[k]$ distort data and how does ideal ZF undo it? | constellations and BER |

Detailed chapter notes are in
[`docs/01_learning_path.md`](docs/01_learning_path.md).

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

The repository is an educational small-scale channel prototype, not an official
3GPP conformance tool and not affiliated with NVIDIA.

## Installation from a clean Python environment

The tested Python versions are 3.11, 3.12, and 3.13. The pinned
`requirements.txt` contains the project itself, NumPy, Matplotlib, CuPy for
CUDA 12, and test/lint tools. No preinstalled Python packages are assumed.

```bash
git clone https://github.com/ThreeMonth03/sionna_tutorial.git
cd sionna_tutorial

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

`requirements.txt` installs `cupy-cuda12x[ctk]`. The `[ctk]` extra installs
CUDA 12 runtime/component wheels into the Python environment, so only a
compatible NVIDIA driver is required. Do not install another `cupy` or
`cupy-cuda*` distribution in the same environment.

Verify the GPU:

```bash
python3 -c "import cupy as cp; print('GPU count:', cp.cuda.runtime.getDeviceCount()); print('GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'])"
```

The RTX 2060 is sufficient for the current examples. `--backend auto` uses CuPy
when a working CUDA installation is available and otherwise falls back to
NumPy.

## Run the examples

```bash
python3 examples/01_single_path.py
python3 examples/02_multipath_mimo.py --backend numpy
python3 examples/03_tdl_profiles.py
python3 examples/04_cdl_channel.py --model C --backend auto
python3 examples/05_mobility_doppler.py --backend auto
python3 examples/06_ofdm_demo.py --backend auto --snr-db 20
```

Figures are written to `artifacts/`.

## Minimal channel-generation API

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
# [batch, rx_antenna, tx_antenna, delayed_path, time]

print(result.delays_s.shape)
# [delayed_path]
```

For an exact CPU/GPU comparison, sample the random state once and reuse it:

```python
state = channel.sample_state(batch_size=128, seed=42)
cpu = channel.generate(128, random_state=state, backend="numpy")
gpu = channel.generate(128, random_state=state, backend="cupy")
```

## Tensor shape map

```text
cluster centers                       [cluster]
ray angles after offsets/coupling     [batch, cluster, ray]
polarization phases                   [batch, cluster, ray, 4]
channel impulse response              [batch, rx, tx, path, time]
OFDM frequency response               [batch, rx, tx, time, subcarrier]
transmitted QPSK                      [batch, tx, subcarrier]
received symbols                      [batch, rx, subcarrier]
```

The deterministic CDL coefficient engine processes one cluster at a time. It
avoids materializing the full

```text
batch x cluster x ray x rx x tx x time
```

tensor while keeping the equations readable.

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

The benchmark includes deterministic coefficient construction and reuses a
fixed random state. It synchronizes CUDA before stopping the timer.

Do not infer that communication simulation is cheap because Examples 01-06
finish quickly. They intentionally use tiny batches and omit coding, channel
estimation, iterative detection/decoding, low-BLER Monte Carlo stopping, and
large parameter sweeps. Likewise, do not infer that it is expensive until a
larger link-level workload is profiled. The current benchmark first answers the
narrower question: how expensive is CDL channel generation as dimensions grow?

## Source reading order

1. [`docs/00_system_overview.md`](docs/00_system_overview.md)
2. [`docs/01_learning_path.md`](docs/01_learning_path.md)
3. [`docs/02_source_map.md`](docs/02_source_map.md)
4. [`docs/algorithm.md`](docs/algorithm.md)
5. `src/sionna_tutorial/profiles.py`
6. `src/sionna_tutorial/rays.py`
7. `src/sionna_tutorial/arrays.py`
8. `src/sionna_tutorial/geometry.py`
9. `src/sionna_tutorial/coefficients.py`
10. `src/sionna_tutorial/cdl.py`
11. `src/sionna_tutorial/ofdm.py`
12. corresponding tests

## References and attribution

- 3GPP TR 38.901, channel models for 0.5-100 GHz
- NVIDIA Sionna, an Apache-2.0 open-source communication-system research library

The packaged profile tables are derived from Sionna's Apache-2.0 model files;
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
