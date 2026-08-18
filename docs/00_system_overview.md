# Chapter 0: Where this repository sits in a communication system

## One-sentence scope

This repository does **not** implement a complete communication system or a
complete 5G NR modem. Its main subject is one block in the middle:

> Generate a complex-baseband wireless propagation channel between a transmit
> antenna array and a receive antenna array.

The main output is the time-varying multipath MIMO channel

```math
h_{u,s}(t,\tau).
```

Here, $s$ identifies a transmit antenna port, $u$ identifies a receive antenna
port, $t$ is time, and $\tau$ is propagation delay.

## The complete end-to-end map

```text
Application data
    |
    v
Transport / network protocols
TCP/UDP, IP
    |
    v
Cellular protocol stack
PDCP, RLC, MAC
    |
    v
+--------------------------- PHY TRANSMITTER ---------------------------+
| CRC -> channel coding -> rate matching -> scrambling -> QAM          |
|     -> layer mapping -> precoding -> resource grid -> OFDM            |
+-----------------------------------------------------------------------+
    |
    v
Digital baseband samples
    |
    v
DAC / RF upconversion / power amplifier
    |
    v
Transmit antenna array
    |
    v
=========================================================================
                 WIRELESS PROPAGATION CHANNEL

     multipath, delay, path power, AoD/AoA, ZoD/ZoA,
     antenna-array phase, polarization, mobility, Doppler

                    THIS REPOSITORY'S CORE
=========================================================================
    |
    v
Receive antenna array
    |
    v
LNA / RF downconversion / ADC
    |
    v
+---------------------------- PHY RECEIVER -----------------------------+
| synchronization -> OFDM FFT -> channel estimation -> MIMO detection  |
|     -> QAM demapping -> channel decoding -> CRC                       |
+-----------------------------------------------------------------------+
    |
    v
MAC, RLC, PDCP, IP, application
```

The code uses a **complex-baseband abstraction**. It does not simulate an
oscillating 3.5 GHz carrier sample by sample, and it does not model RF circuit
nonidealities. Carrier frequency still matters because it determines
wavelength and therefore antenna spatial phase and Doppler.

## Current implementation coverage

| System block | Covered? | Where |
|---|---:|---|
| TCP/IP, PDCP, RLC, MAC | No | Outside this repository |
| CRC, LDPC, rate matching, HARQ | No | Outside this repository |
| QAM mapping | Minimal | Example 06 uses uncoded QPSK |
| OFDM resource grid, IFFT, cyclic prefix | No | Example 06 is frequency-domain only |
| Precoding / beamforming | No | Not yet implemented |
| RF electronics | No | Ideal complex-baseband abstraction |
| Antenna-array geometry | Yes | `arrays.py` |
| TDL/CDL propagation profile | Yes | `profiles.py`, `tdl.py`, `cdl.py` |
| Multipath clusters and rays | Yes | `rays.py`, `coefficients.py` |
| Polarization and array phase | Yes | `arrays.py`, `coefficients.py` |
| Mobility and Doppler | Yes | `geometry.py`, `coefficients.py` |
| Channel application | Minimal | `ofdm.py`, Example 06 |
| Pilot / DMRS | No | Not yet implemented |
| Channel estimation | No | Example 06 is given the exact channel |
| MIMO equalization | Minimal | Perfect-CSI pseudo-inverse in Example 06 |
| Soft demapping and decoding | No | Hard QPSK decisions only |
| BER / BLER Monte Carlo campaign | Minimal | One small BER demonstration only |

## Three different operations that all contain the word "channel"

These are easy to confuse, but they are different parts of a simulator.

### 1. Channel generation

Generate a mathematical description of the propagation environment:

```python
realization = channel.generate(...)
```

Output:

```text
coefficients  [batch, rx_antenna, tx_antenna, path, time]
delays        [path]
```

This is what Examples 01-05 primarily study.

### 2. Channel application

Take transmitted data $\mathbf{x}$ and pass it through a known channel
$\mathbf{H}$:

```math
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}.
```

For a frequency-selective time-domain channel, the more general expression is

```math
y_u(t)
=
\sum_s\int h_{u,s}(t,\tau)x_s(t-\tau)\,d\tau+n_u(t).
```

Example 06 performs a simplified frequency-domain channel application on OFDM
subcarriers.

### 3. Channel estimation

A real receiver does not know the true $\mathbf{H}$. It transmits known pilots,
observes how they were distorted, and estimates

```math
\widehat{\mathbf{H}}.
```

Only then can the receiver equalize or detect the unknown data. This repository
does not yet implement channel estimation. Example 06 directly gives the true
simulated $\mathbf{H}$ to the equalizer; this idealization is called
**perfect CSI**.

## The main mathematical object

The channel generator returns a sparse channel impulse response:

```math
h_{u,s}(t,\tau)
=
\sum_n h_{u,s,n}(t)\,\delta(\tau-\tau_n).
```

Interpretation:

- $n$ selects a delayed propagation cluster/path,
- $\tau_n$ is when that path arrives,
- $h_{u,s,n}(t)$ is its complex gain between Tx port $s$ and Rx port $u$,
- magnitude describes attenuation,
- phase describes propagation and antenna-array phase,
- time variation describes Doppler fading.

For OFDM, the delayed paths are converted into one matrix per subcarrier:

```math
H_{u,s}[k,t]
=
\sum_n h_{u,s,n}(t)e^{-j2\pi f_k\tau_n}.
```

Then the familiar narrowband MIMO equation is applied independently on each
subcarrier:

```math
\mathbf{y}[k]
=
\mathbf{H}[k]\mathbf{x}[k]+\mathbf{n}[k].
```

## TDL and CDL in the system map

Both TDL and CDL are models for the **wireless propagation channel**.

### TDL: tapped delay line

TDL primarily specifies a power-delay profile:

```text
delay 0 ns    -> power -2 dB
delay 40 ns   -> power -5 dB
delay 130 ns  -> power -10 dB
```

It answers: how many delayed echoes arrive, when do they arrive, and how strong
are they on average?

### CDL: clustered delay line

CDL adds spatial MIMO structure. Every delayed cluster contains multiple rays,
and every ray has departure/arrival angles, polarization, phase, and Doppler.
It answers an additional question: how does the same propagation environment
appear across all Tx-Rx antenna pairs?

```text
cluster delay/power
    -> 20 rays
    -> AoD/AoA/ZoD/ZoA
    -> antenna field and polarization
    -> array-position phase
    -> Doppler phase
    -> sum rays
    -> one delayed MIMO path matrix
```

## Tensor data flow

```text
CDL profile tables
  delays, powers, cluster-center angles
  [cluster]
          |
          v
Sample random per-ray state
  angles                   [batch, cluster, ray]
  polarization phases      [batch, cluster, ray, 4]
          |
          v
Generate channel coefficients
  h                        [batch, rx, tx, path, time]
  tau                      [path]
          |
          v
CIR -> OFDM frequency response
  H                        [batch, rx, tx, time, subcarrier]
          |
          v
Generate QPSK data
  X                        [batch, tx, subcarrier]
          |
          v
Apply channel
  Y = H X + N              [batch, rx, subcarrier]
          |
          v
Perfect-CSI equalization
  X_hat                    [batch, tx, subcarrier]
```

## How the six examples fit together

```text
Example 01: one ray
    spatial phase + Doppler
            |
            v
Example 02: many rays and antennas
    rays -> one MIMO matrix H
            |
            +----------------------+
            |                      |
            v                      v
Example 03: delayed echoes      spatial structure
    TDL power-delay profile        |
            |                      |
            +----------+-----------+
                       v
Example 04: complete CDL realization
    h(rx, tx, path, time)
                       |
                       v
Example 05: observe H changing with mobility
                       |
                       v
Example 06: put H inside a tiny frequency-domain link
    QPSK -> H[k] -> noise -> perfect-CSI ZF -> BER
```

Examples 01-03 are prerequisites. Example 04 is the main channel-generation
example. Example 05 studies time variation. Example 06 is the first,
deliberately small step from channel modeling toward a complete PHY link.

## What a complete 5G link would add next

A more realistic link-level simulator would eventually add:

```text
information bits
-> CRC and LDPC
-> rate matching and scrambling
-> QAM
-> layer mapping and precoding
-> OFDM resource grid and DMRS pilots
-> time-domain OFDM waveform
-> TDL/CDL channel
-> synchronization and FFT
-> channel estimation H_hat
-> LMMSE/K-Best detection
-> soft demapping to LLRs
-> LDPC decoding
-> CRC and BLER statistics
```

That larger chain is intentionally not hidden inside the current tutorial. The
present repository first makes the channel block understandable and testable.

## Continue reading

1. [`01_learning_path.md`](01_learning_path.md): what each example teaches.
2. [`02_source_map.md`](02_source_map.md): how files and function calls connect.
3. [`algorithm.md`](algorithm.md): formulas used inside the channel generator.
