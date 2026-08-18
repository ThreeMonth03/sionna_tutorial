# From a propagation ray to a MIMO channel tensor

This note follows the order used by the source code. It is intentionally a
numerical explanation, not a replacement for TR 38.901.

## 1. What the channel model returns

The CDL engine returns

\[
h[b,u,s,n,t],
\]

where

- `b` is an independent Monte Carlo realization,
- `u` is a receive antenna port,
- `s` is a transmit antenna port,
- `n` is a delayed path/cluster,
- `t` is a sampled time.

The corresponding delay of path `n` is `tau[n]`. Together they describe the
sparse channel impulse response

\[
h_{u,s}(t,\tau)=\sum_n h_{u,s,n}(t)\,\delta(\tau-\tau_n).
\]

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

\[
K = \frac{P_{\mathrm{LOS}}}{\sum P_{\mathrm{diffuse}}}.
\]

## 3. Twenty rays inside one cluster

One cluster is not one plane wave. CDL expands every diffuse cluster into 20
rays. TR 38.901 Table 7.5-3 provides the normalized offsets

\[
\alpha_m \in
\{\pm0.0447,\pm0.1413,\ldots,\pm2.1551\}.
\]

For example, the azimuth of arrival of ray `m` in cluster `n` is

\[
\phi_{n,m,\mathrm{AOA}}
=
\phi_{n,\mathrm{AOA}} + c_{\mathrm{ASA}}\alpha_m.
\]

The same construction is applied to AoD, ZoA, and ZoD. Their ray orders are
then independently shuffled. This is random coupling: one ray does not always
receive the same offset in all four angular dimensions.

See `constants.py` and `rays.py`.

## 4. Angles become direction vectors

Zenith `theta` and azimuth `phi` become the Cartesian unit vector

\[
\hat{r}(\theta,\phi)=
\begin{bmatrix}
\sin\theta\cos\phi\\
\sin\theta\sin\phi\\
\cos\theta
\end{bmatrix}.
\]

See `unit_sphere_vector()` in `geometry.py`.

## 5. Antenna-position phase

A plane wave reaches different array elements at different phases. For an
antenna at position `d`, the spatial term is

\[
\exp\left(j\frac{2\pi}{\lambda}\hat{r}^{T}d\right).
\]

A ULA/UPA is therefore not only a list of signal values: its physical element
positions are part of the channel equation. See `AntennaArray` and
`spatial_phase()`.

## 6. Polarization coupling

Each ray has four random initial phases and a cross-polarization power ratio
`kappa`. They form a 2×2 matrix

\[
\mathbf{P}_{n,m}=
\begin{bmatrix}
e^{j\Phi_{\theta\theta}} &
\kappa^{-1/2}e^{j\Phi_{\theta\phi}}\\
\kappa^{-1/2}e^{j\Phi_{\phi\theta}} &
e^{j\Phi_{\phi\phi}}
\end{bmatrix}.
\]

The receive field vector, this matrix, and the transmit field vector are
contracted:

\[
\mathbf{F}_{rx}^{T}\mathbf{P}_{n,m}\mathbf{F}_{tx}.
\]

Dual-polarized arrays are represented as two antenna ports at the same physical
location with -45° and +45° slants. See `_polarization_matrix()` and
`element_field_components()`.

## 7. Doppler phase

For a moving endpoint with velocity vector `v`, a ray's Doppler frequency is

\[
\nu_{n,m}=\frac{\hat{r}_{n,m}^{T}v}{\lambda}.
\]

Its time evolution is

\[
\exp(j2\pi\nu_{n,m}t).
\]

At zero velocity every coefficient remains constant over time. Increasing
speed increases the rate of phase rotation and fading. See `doppler_phase()`.

## 8. Sum rays into one cluster

For diffuse cluster power `P_n` and 20 rays, each ray receives amplitude
`√(P_n/20)`. The code multiplies

```text
receive field
× polarization matrix
× transmit field
× receive array phase
× transmit array phase
× Doppler phase
× sqrt(cluster power / 20)
```

and sums the ray axis. The result is one tensor

```text
[batch, receive antenna, transmit antenna, time]
```

for that cluster. The implementation loops over clusters so it does not create
a huge tensor containing every cluster, ray, antenna pair, and time sample at
once. See `_cluster_coefficients()`.

## 9. Add the LOS component

For CDL-D/E, diffuse coefficients are scaled by

\[
\sqrt{\frac{1}{K+1}},
\]

and the deterministic LOS component by

\[
\sqrt{\frac{K}{K+1}}.
\]

The LOS component is added to the first zero-delay diffuse cluster. See
`_los_coefficient()` and the final block of `generate_cdl_coefficients()`.

## 10. Convert CIR to an OFDM channel

The response on baseband subcarrier frequency `f_k` is

\[
H[k,t]=\sum_n h[n,t]e^{-j2\pi f_k\tau_n}.
\]

This transform is implemented by `cir_to_frequency_response()`. The minimal
OFDM demo then applies

\[
y[k]=H[k]x[k]+n[k]
\]

and uses a perfect-CSI pseudo-inverse to recover `x[k]`.

## Where to inspect performance later

The readable prototype exposes natural candidates without optimizing them yet:

- trigonometric/exponential evaluation for rays,
- antenna field responses reused across batches,
- polarization contractions,
- ray-to-cluster reduction,
- CIR-to-CFR transformation,
- batched small MIMO pseudo-inverses.

The benchmark exists to determine whether any of these are actually important
on the target CPU/GPU before writing specialized kernels.
