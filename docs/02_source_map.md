# Source map: how the code fits together

Read [`00_system_overview.md`](00_system_overview.md) before this file. The
source tree implements the wireless-channel block, plus one tiny
frequency-domain application of that channel.

## Public call flow

The most important call chain is:

```text
CDLChannel(...)
    |
    +-- load_cdl_profile()
    |       standardized delays, powers, angles, spreads, XPR, LOS data
    |
    +-- sample_state()
    |       sample_cdl_random_state()
    |       per-ray angles, random coupling, polarization phases
    |
    `-- generate()
            |
            `-- generate_cdl_coefficients()
                    |
                    +-- for every cluster:
                    |       _cluster_coefficients()
                    |       20 rays -> one delayed MIMO path
                    |
                    +-- optional _los_coefficient() for CDL-D/E
                    |
                    `-- ChannelCoefficients
                            coefficients [batch, rx, tx, path, time]
                            delays_s    [path]

ChannelCoefficients
    |
    `-- cir_to_frequency_response()
            H [batch, rx, tx, time, subcarrier]
                    |
                    `-- run_perfect_csi_ofdm_demo()
                            QPSK -> H X + N -> pseudo-inverse -> BER
```

## File-by-file map

### `profiles.py`: standardized average channel profiles

**Role in the system**

Defines the statistical environment before one random channel realization is
sampled.

**Input**

- model letter A-E,
- requested RMS delay spread.

**Output**

- path/cluster delays,
- normalized average powers,
- mean departure and arrival angles,
- angular spreads,
- XPR,
- LOS/K-factor metadata.

**Does not do**

It does not generate antenna coefficients or transmitted/received signals.

---

### `rays.py`: one random small-scale channel state

**Role in the system**

Expands each CDL cluster into 20 physical rays and samples the random variables
that differ between Monte Carlo realizations.

**Input**

A deterministic `CDLProfile`, batch size, and random seed.

**Output**

- per-ray AoA/AoD/ZoA/ZoD,
- independently shuffled angular coupling,
- four polarization phases per ray.

The returned `CDLRandomState` is NumPy data. It can be reused by both the NumPy
and CuPy implementations for exact differential testing.

---

### `arrays.py`: antenna positions and element fields

**Role in the system**

Describes the physical antenna ports at the two ends of the propagation
channel.

**Contains**

- ULA and UPA element positions,
- single- and dual-polarized ports,
- polarization slant angles,
- omni and simplified TR 38.901-style element patterns.

Antenna positions are measured in meters. The channel formula converts them to
phase using wavelength and ray direction.

---

### `geometry.py`: angle, spatial phase, and Doppler primitives

**Role in the system**

Converts geometrical quantities into complex phases.

**Important functions**

- `unit_sphere_vector(theta, phi)`: angles -> Cartesian direction,
- `spatial_phase(position, direction, wavelength)`: array-position phase,
- `doppler_phase(direction, velocity, time, wavelength)`: mobility phase.

These functions explain most of Example 01.

---

### `coefficients.py`: deterministic channel-coefficient engine

**Role in the system**

This is the mathematical core. It takes a profile, sampled ray state, antenna
arrays, velocity, and time samples and generates

```text
h [batch, rx, tx, path, time].
```

For each ray it combines:

```text
receive antenna field
x polarization matrix
x transmit antenna field
x receive array-position phase
x transmit array-position phase
x Doppler phase
x sqrt(cluster power / 20)
```

It then sums the 20-ray axis to produce one delayed MIMO path per cluster.

**Important functions**

- `_polarization_matrix()`
- `_cluster_coefficients()`
- `_los_coefficient()`
- `generate_cdl_coefficients()`

**Why it loops over clusters**

The readable implementation avoids constructing the complete

```text
batch x cluster x ray x rx x tx x time
```

tensor at once. This reduces peak memory and keeps one cluster's equation
visible.

---

### `cdl.py`: high-level channel-generation API

**Role in the system**

Connects profile loading, random-state sampling, backend selection, and the
deterministic coefficient engine.

`CDLChannel.generate()` generates a propagation channel. It does not transmit
bits, estimate the channel, or decode data.

The separation

```text
sample random state -> deterministic coefficient calculation
```

is intentional. It permits reproducibility and fair NumPy/CuPy comparison.

---

### `tdl.py`: simpler delay-tap channel

**Role in the system**

Provides a less spatially detailed channel model based on standardized tap
delays and powers. It is useful for understanding delay spread before studying
CDL ray geometry.

The tutorial uses a compact sum-of-sinusoids time variation. It is educational,
not a claim of complete 3GPP conformance.

---

### `ofdm.py`: one minimal application of the generated channel

**Role in the system**

Moves one step beyond channel generation.

1. `cir_to_frequency_response()` converts delayed paths $h(t,\tau)$ into one
   MIMO matrix $\mathbf{H}[k,t]$ per subcarrier.
2. `run_perfect_csi_ofdm_demo()` generates uncoded QPSK symbols.
3. It applies $\mathbf{Y}[k] = \mathbf{H}[k]\mathbf{X}[k] + \mathbf{N}[k]$.
4. It gives the exact $\mathbf{H}[k]$ to a pseudo-inverse equalizer.
5. It makes hard QPSK decisions and counts bit errors.

**What the name does not imply**

This is not a complete OFDM waveform implementation. There is no time-domain
IFFT, cyclic prefix, synchronization, pilot, DMRS, or channel estimator.

---

### `backend.py`: NumPy/CuPy selection

**Role in the system**

Provides the small compatibility layer used by the same equations on CPU or
CUDA GPU.

It is not a scheduler and does not split one computation across devices.

---

### `tests/`: physical and numerical invariants

The tests are part of the learning material. They show which properties should
remain true:

- output shapes and dtypes,
- reproducibility for a fixed state,
- a zero-velocity channel is constant over time,
- a moving channel changes over time,
- average diffuse power is normalized,
- TDL/CDL tables load correctly,
- CIR-to-CFR shapes are correct,
- NumPy and CuPy agree for the same random state.

## Data ownership and transformations

```text
profile tables
    persistent, small, CPU-resident
        |
        v
CDLRandomState
    sampled once on CPU, reusable and reproducible
        |
        v
backend.asarray(...)
    move the selected cluster's state to NumPy or CuPy arrays
        |
        v
ChannelCoefficients
    large numerical output on the selected backend
        |
        v
backend.to_numpy(...)
    only for plotting, scalar reporting, or CPU/GPU comparison
```

During performance work, avoid calling `to_numpy()` inside the channel's hot
path because it synchronizes and transfers data back to the CPU.

## Where future full-link blocks would connect

```text
future transmitter
bits -> LDPC -> QAM -> resource grid -> OFDM
                                      |
                                      v
                           ChannelCoefficients / H[k]
                                      |
                                      v
future receiver
FFT -> pilot extraction -> channel estimation -> detection -> demap -> LDPC
```

The current channel API can remain useful when those blocks are added. The
channel generator itself does not need to know about TCP/IP, MAC scheduling, or
LDPC decoding.
