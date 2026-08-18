# Learning path: from one ray to a tiny OFDM link

Read [`00_system_overview.md`](00_system_overview.md) first. It places this
repository inside the complete communication stack.

The examples are deliberately small. Their purpose is to isolate one idea at a
time, not to measure the runtime of a production communication simulator.

## Chapter 1: One propagation ray

**Run**

```bash
python3 examples/01_single_path.py
```

**System location**

```text
Tx antenna -> [one ray inside the wireless channel] -> Rx antenna
```

**Question**

Why does one plane wave have a different complex phase at each antenna, and why
does motion rotate that phase over time?

**Inputs**

- carrier frequency and wavelength,
- a four-element Tx uniform linear array,
- one propagation direction,
- one velocity vector,
- sampled times.

**Output**

A complex coefficient for every antenna and time sample:

```text
[tx_antenna, time]
```

The figure plots the real part and unwrapped phase.

**Core code**

- `unit_sphere_vector()` in `geometry.py`
- `spatial_phase()` in `geometry.py`
- `doppler_phase()` in `geometry.py`
- `AntennaArray.ula()` in `arrays.py`

**Not modeled**

No multipath, no receive array, no delay, no noise, no transmitted data, and no
receiver.

**Checkpoint**

You should be able to explain why antenna position changes phase even when all
antennas receive the same physical ray.

---

## Chapter 2: Many rays become one MIMO matrix

**Run**

```bash
python3 examples/02_multipath_mimo.py --backend numpy
```

**System location**

```text
Tx antenna array -> [multipath spatial channel] -> Rx antenna array
```

**Question**

Where does the matrix $\mathbf{H}$ in

```math
\mathbf{y}=\mathbf{H}\mathbf{x}+\mathbf{n}
```

come from physically?

**Inputs**

- CDL-A profile,
- eight Tx antenna ports,
- four Rx antenna ports,
- sampled ray angles, phases, and polarization.

**Output**

The example sums all delayed paths for one instant and shows

```text
H [rx_antenna=4, tx_antenna=8]
```

as magnitude and phase heatmaps.

**Core code**

- `CDLChannel.generate()` in `cdl.py`
- `generate_cdl_coefficients()` in `coefficients.py`

**Important simplification**

Summing all delays into one matrix is only for visualization. A
frequency-selective receiver normally preserves the delayed paths or converts
them to one $\mathbf{H}[k]$ per OFDM subcarrier.

**Checkpoint**

You should be able to explain why each Tx-Rx antenna pair has a different
complex coefficient.

---

## Chapter 3: Multipath in the delay domain

**Run**

```bash
python3 examples/03_tdl_profiles.py
```

**System location**

```text
Wireless channel -> delayed echoes
```

**Question**

Why is a wireless channel not described by only one matrix at one delay?

**Inputs**

The standardized TDL-A/B/C/D/E delay and average-power tables, scaled to a
100 ns RMS delay spread.

**Output**

A power-delay profile for each TDL model.

**Core code**

- `load_tdl_profile()` in `profiles.py`
- packaged JSON profile tables

**Important clarification**

This example mostly loads and plots standard tables. It is not a heavy Monte
Carlo simulation.

**Checkpoint**

You should be able to explain frequency-selective fading: different delayed
echoes add with different phases on different frequencies.

---

## Chapter 4: A 3GPP CDL MIMO channel realization

**Run**

```bash
python3 examples/04_cdl_channel.py --model C --backend numpy
```

**System location**

```text
Tx array -> [3GPP clustered MIMO propagation channel] -> Rx array
```

**Question**

How are delay, cluster power, 20 rays, four angular dimensions, antenna
patterns, polarization, array geometry, and random phase combined into
$h(t,\tau)$?

**Inputs**

- one CDL-A/B/C/D/E profile,
- delay spread and carrier frequency,
- dual-polarized Tx planar array,
- single-polarized Rx linear array,
- random channel state.

**Output**

```text
coefficients [batch=1, rx, tx, path, time=1]
delays       [path]
```

The figure shows realized path powers and one instantaneous MIMO magnitude
matrix.

**Core code**

Read in this order:

1. `profiles.py`
2. `rays.py`
3. `arrays.py`
4. `geometry.py`
5. `coefficients.py`
6. `cdl.py`

**Not modeled**

No path loss, geographical layout, UMi/UMa/RMa scenario, RF hardware, data
symbols, receiver, or channel estimation.

**Checkpoint**

You should be able to describe the path

```text
standard cluster -> 20 rays -> per-ray coefficient -> ray sum -> delayed MIMO path
```

and identify the tensor axes in the returned result.

---

## Chapter 5: Mobility and Doppler fading

**Run**

```bash
python3 examples/05_mobility_doppler.py --backend numpy
```

**System location**

```text
Time-varying wireless channel H(t, tau)
```

**Question**

Why does a faster terminal make the channel change faster?

**Inputs**

The same sampled CDL-B ray geometry is reused at 0, 30, and 120 km/h. Reusing
the random state isolates the effect of velocity.

**Output**

One narrowband composite channel trace over 100 ms for each speed.

**Core code**

- `doppler_phase()` in `geometry.py`
- `_cluster_coefficients()` in `coefficients.py`

**Important simplification**

The example uses one Tx and one Rx antenna and sums all paths. It is designed to
show Doppler intuition, not to reproduce a full receiver.

**Checkpoint**

You should be able to relate speed, wavelength, Doppler frequency, and channel
coherence time.

---

## Chapter 6: Put the channel inside a tiny link

**Run**

```bash
python3 examples/06_ofdm_demo.py --backend numpy --snr-db 20
```

**System location**

```text
QPSK data -> OFDM subcarriers -> channel application -> ideal equalizer -> bits
```

This is the only example that crosses from a transmitter, through the channel,
to a receiver operation.

**Question**

How does the generated $h(t,\tau)$ distort data, and how can a receiver undo a
known MIMO channel?

**Inputs**

- 64 independent channel realizations,
- two Tx and four Rx antennas,
- 64 frequency-domain OFDM subcarriers,
- uncoded QPSK,
- CDL-C channel,
- additive Gaussian noise.

**Output**

- transmitted QPSK constellation,
- one received-antenna constellation,
- equalized constellation,
- hard-decision BER.

**Core code**

- `cir_to_frequency_response()` in `ofdm.py`
- `run_perfect_csi_ofdm_demo()` in `ofdm.py`

**The largest idealization: perfect CSI**

The equalizer receives the exact simulated channel $\mathbf{H}[k]$. A real
receiver must insert pilots such as DMRS, estimate
$\widehat{\mathbf{H}}[k]$, and equalize using that imperfect estimate.

**Other missing parts**

- no IFFT or cyclic prefix,
- no synchronization,
- no resource grid,
- no precoding,
- no coding or LDPC,
- no soft demapping,
- no HARQ,
- no BLER campaign.

**Checkpoint**

You should be able to distinguish channel generation, channel application, and
channel estimation.

---

## After the six examples

Read the implementation in this order:

1. [`02_source_map.md`](02_source_map.md)
2. [`algorithm.md`](algorithm.md)
3. `src/sionna_tutorial/profiles.py`
4. `src/sionna_tutorial/rays.py`
5. `src/sionna_tutorial/arrays.py`
6. `src/sionna_tutorial/geometry.py`
7. `src/sionna_tutorial/coefficients.py`
8. `src/sionna_tutorial/cdl.py`
9. `src/sionna_tutorial/ofdm.py`
10. corresponding tests

Do not use the runtime of Examples 01-06 to judge whether communication
simulation is computationally expensive. They intentionally use tiny batches
and isolate concepts. `benchmarks/benchmark_cdl.py` is the first scaling tool;
a future coded link-level Monte Carlo workload would be required to evaluate a
complete communication-simulation bottleneck.
