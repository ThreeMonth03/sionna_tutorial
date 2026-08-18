# Inside the channel generator: from a propagation ray to a MIMO tensor

This document explains the numerical algorithm **inside the wireless-channel
block**. It is not an end-to-end communication-system description and not a
replacement for TR 38.901.

Read these first:

1. [`00_system_overview.md`](00_system_overview.md) — where the channel sits in
   the complete transmitter/channel/receiver chain.
2. [`01_learning_path.md`](01_learning_path.md) — how Examples 01-06 build the
   concepts.
3. [`02_source_map.md`](02_source_map.md) — file roles and function call graph.

## 0. Inputs, outputs, and call graph

The channel engine does not take information bits as input. It takes a
statistical propagation profile and physical configuration:

```text
CDL model A-E
RMS delay spread
carrier frequency
Tx/Rx antenna arrays
velocity and sampled times
random seed/state
```

and returns the channel itself:

```text
h      [batch, rx_antenna, tx_antenna, delayed_path, time]
tau    [delayed_path]
```

The high-level call path is:

```text
CDLChannel.__post_init__()
    -> load_cdl_profile()

CDLChannel.sample_state()
    -> sample_cdl_random_state()

CDLChannel.generate()
    -> generate_cdl_coefficients()
        -> _cluster_coefficients() for every diffuse cluster
        -> _los_coefficient() for CDL-D/E
    -> ChannelCoefficients

ChannelCoefficients
    -> cir_to_frequency_response()
    -> optional tiny channel application in Example 06
```

The separation between random-state sampling and deterministic coefficient
calculation is intentional. The same physical random state can be executed on
NumPy and CuPy for exact CPU/GPU comparison.

## 1. What the channel model returns

The CDL engine returns

```math
h[b,u,s,n,t].
```

Here,

- $b$ is an independent Monte Carlo realization,
- $u$ is a receive antenna port,
- $s$ is a transmit antenna port,
- $n$ is a delayed path/cluster,
- $t$ is a sampled time.

The corresponding delay of path $n$ is $\tau_n$. Together they describe the
sparse channel impulse response

```math
h_{u,s}(t,\tau)=\sum_n h_{u,s,n}(t)\,\delta(\tau-\tau_n).
```

This is **channel generation**. Transmitting data through this channel and
estimating the channel at a receiver are separate operations.

See `ChannelCoefficients` in `coefficients.py`.

## 2. Standardized cluster profiles

A CDL profile supplies, for every cluster:

- normalized delay,
- average power,
- mean AoD and AoA,
- mean ZoD and ZoA.

It also supplies cluster-wise angular spreads and a cross-polarization ratio.
`profiles.py` loads these tables, converts dB powers to linear power, normalizes
them, and scales normalized delays by the requested RMS delay spread.

For LOS models D/E, the first table entry is the specular component. The code
extracts it and computes

```math
K = \frac{P_{\mathrm{LOS}}}{\sum P_{\mathrm{diffuse}}}.
```

At this stage there is still no antenna-pair coefficient. The profile is an
average statistical description of the environment.

## 3. Twenty rays inside one cluster

One cluster is not one plane wave. CDL expands every diffuse cluster into 20
rays. TR 38.901 Table 7.5-3 provides the normalized offsets

```math
\alpha_m \in
\{\pm0.0447,\pm0.1413,\ldots,\pm2.1551\}.
```

For example, the azimuth of arrival of ray $m$ in cluster $n$ is

```math
\phi_{n,m,\mathrm{AOA}}
=
\phi_{n,\mathrm{AOA}} + c_{\mathrm{ASA}}\alpha_m.
```

The same construction is applied to AoD, ZoA, and ZoD. Their ray orders are
then independently shuffled. This is random coupling: one ray does not always
receive the same offset in all four angular dimensions.

The important tensor shape is

```text
[batch, cluster, ray]
```

for each angular quantity.

See `constants.py` and `rays.py`.

## 4. Angles become direction vectors

Zenith $\theta$ and azimuth $\phi$ become the Cartesian unit vector

```math
\hat{\mathbf{r}}(\theta,\phi)=
\begin{bmatrix}
\sin\theta\cos\phi\\
\sin\theta\sin\phi\\
\cos\theta
\end{bmatrix}.
```

This vector connects the communication meaning of AoA/AoD to ordinary geometry
and dot products.

See `unit_sphere_vector()` in `geometry.py`.

## 5. Antenna-position phase

A plane wave reaches different array elements at different phases. For an
antenna at position $\mathbf{d}$, the spatial term is

```math
\exp\left(j\frac{2\pi}{\lambda}\hat{\mathbf{r}}^{T}\mathbf{d}\right).
```

A ULA/UPA is therefore not only a list of signal values: its physical element
positions are part of the channel equation. The spatial phase is what turns
one propagation ray into a matrix over all Tx-Rx antenna pairs.

See `AntennaArray` and `spatial_phase()`.

## 6. Polarization coupling

Each ray has four random initial phases and a cross-polarization power ratio
$\kappa$. They form a $2\times2$ matrix

```math
\mathbf{P}_{n,m}=
\begin{bmatrix}
e^{j\Phi_{\theta\theta}} &
\kappa^{-1/2}e^{j\Phi_{\theta\phi}}\\
\kappa^{-1/2}e^{j\Phi_{\phi\theta}} &
e^{j\Phi_{\phi\phi}}
\end{bmatrix}.
```

The receive field vector, this matrix, and the transmit field vector are
contracted:

```math
\mathbf{F}_{\mathrm{rx}}^{T}\mathbf{P}_{n,m}\mathbf{F}_{\mathrm{tx}}.
```

Dual-polarized arrays are represented as two antenna ports at the same physical
location with $-45^\circ$ and $+45^\circ$ slants. See
`_polarization_matrix()` and `element_field_components()`.

## 7. Doppler phase

For a moving endpoint with velocity vector $\mathbf{v}$, a ray's Doppler
frequency is

```math
f_{D,n,m}=\frac{\hat{\mathbf{r}}_{n,m}^{T}\mathbf{v}}{\lambda}.
```

Its time evolution is

```math
\exp(j2\pi f_{D,n,m}t).
```

At zero velocity every coefficient remains constant over time. Increasing
speed increases the rate of phase rotation and fading. Example 05 reuses the
same random rays at multiple speeds so only this factor changes.

See `doppler_phase()`.

## 8. The per-ray product and ray reduction

For diffuse cluster power $P_n$ and 20 rays, each ray receives amplitude
$\sqrt{P_n/20}$. For one ray and one antenna pair, the implementation combines

```text
receive antenna field
x polarization matrix
x transmit antenna field
x receive array-position phase
x transmit array-position phase
x Doppler phase
x sqrt(cluster power / 20)
```

The corresponding source sequence inside `_cluster_coefficients()` is:

```text
ray angles
    -> tx_direction / rx_direction
    -> tx_field / rx_field
    -> polarization matrix
    -> field_coupling
    -> tx_spatial / rx_spatial
    -> Doppler time_phase
    -> multiply factors
    -> sum the 20-ray axis
```

The output for one cluster is

```text
[batch, receive_antenna, transmit_antenna, time].
```

The implementation loops over clusters so it does not create a huge tensor
containing every cluster, ray, antenna pair, and time sample at once. See
`_cluster_coefficients()`.

## 9. Add the LOS component

For CDL-D/E, diffuse coefficients are scaled by

```math
\sqrt{\frac{1}{K+1}},
```

and the deterministic LOS component by

```math
\sqrt{\frac{K}{K+1}}.
```

The LOS component is added to the first zero-delay diffuse cluster. See
`_los_coefficient()` and the final block of `generate_cdl_coefficients()`.

At the end of this step, the channel generator has completed its task. The
result is $h(t,\tau)$, not transmitted or detected user data.

## 10. Convert CIR to an OFDM channel

For a baseband subcarrier frequency $f_k$, the response is

```math
H[k,t]=\sum_n h[n,t]e^{-j2\pi f_k\tau_n}.
```

This transform is implemented by `cir_to_frequency_response()`. It changes the
shape from

```text
[batch, rx, tx, path, time]
```

to

```text
[batch, rx, tx, time, subcarrier].
```

The minimal OFDM demo then applies

```math
y[k]=H[k]x[k]+n[k]
```

and uses a perfect-CSI pseudo-inverse to recover $x[k]$.

Important: Example 06 is frequency-domain only. It does not implement an IFFT,
cyclic prefix, timing synchronization, pilot/DMRS, or channel estimation.

## Where to inspect performance later

The readable prototype exposes natural candidates without assuming they are
real bottlenecks:

- trigonometric/exponential evaluation for rays,
- antenna field responses reused across batches,
- polarization contractions,
- ray-to-cluster reduction,
- CIR-to-CFR transformation,
- batched small MIMO pseudo-inverses.

Examples 01-06 intentionally use tiny dimensions and should finish quickly.
`benchmarks/benchmark_cdl.py` scales channel generation. A complete coded
link-level Monte Carlo workload would be required before drawing conclusions
about the dominant cost of a full communication simulator.
