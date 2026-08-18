from __future__ import annotations

import numpy as np
import pytest

from sionna_tutorial import AntennaArray, CDLChannel, cupy_available, get_backend


def _channel(model: str = "A", precision: str = "single") -> CDLChannel:
    fc = 3.5e9
    return CDLChannel(
        model=model,  # type: ignore[arg-type]
        delay_spread_s=100e-9,
        carrier_frequency_hz=fc,
        tx_array=AntennaArray.ula(2, fc),
        rx_array=AntennaArray.ula(3, fc),
        precision=precision,  # type: ignore[arg-type]
    )


def test_cdl_shapes_dtype_and_reproducibility() -> None:
    channel = _channel()
    state = channel.sample_state(4, seed=5)
    one = channel.generate(
        4,
        num_time_steps=6,
        sampling_frequency_hz=2_000.0,
        velocity_mps=(10.0, 0.0, 0.0),
        random_state=state,
        backend="numpy",
    )
    two = channel.generate(
        4,
        num_time_steps=6,
        sampling_frequency_hz=2_000.0,
        velocity_mps=(10.0, 0.0, 0.0),
        random_state=state,
        backend="numpy",
    )
    assert one.coefficients.shape == (4, 3, 2, 23, 6)
    assert one.coefficients.dtype == np.complex64
    np.testing.assert_array_equal(one.coefficients, two.coefficients)


def test_zero_velocity_is_constant_over_time() -> None:
    channel = _channel()
    result = channel.generate(
        2,
        num_time_steps=4,
        sampling_frequency_hz=1_000.0,
        velocity_mps=(0.0, 0.0, 0.0),
        seed=7,
        backend="numpy",
    )
    np.testing.assert_allclose(result.coefficients[..., 0], result.coefficients[..., -1])


def test_nonzero_velocity_changes_channel() -> None:
    channel = _channel()
    result = channel.generate(
        2,
        num_time_steps=8,
        sampling_frequency_hz=1_000.0,
        velocity_mps=(25.0, 0.0, 0.0),
        seed=7,
        backend="numpy",
    )
    assert not np.allclose(result.coefficients[..., 0], result.coefficients[..., -1])


def test_average_nlos_path_energy_is_normalized() -> None:
    fc = 3.5e9
    channel = CDLChannel(
        "A",
        100e-9,
        fc,
        AntennaArray.ula(1, fc),
        AntennaArray.ula(1, fc),
        precision="double",
    )
    result = channel.generate(2_000, seed=10, backend="numpy")
    energy = np.mean(np.sum(np.abs(result.coefficients[..., 0]) ** 2, axis=-1))
    assert energy == pytest.approx(1.0, rel=0.08)


def test_los_models_keep_standard_cluster_count() -> None:
    result = _channel("D").generate(2, seed=0, backend="numpy")
    assert result.coefficients.shape[3] == 13
    assert result.delays_s[0] == 0.0


@pytest.mark.cuda
@pytest.mark.skipif(not cupy_available(), reason="working CuPy/CUDA installation not available")
def test_numpy_and_cupy_match_for_same_random_state() -> None:
    channel = _channel("B")
    state = channel.sample_state(3, seed=123)
    cpu = channel.generate(3, num_time_steps=3, random_state=state, backend="numpy")
    gpu_backend = get_backend("cupy")
    gpu = channel.generate(3, num_time_steps=3, random_state=state, backend=gpu_backend)
    np.testing.assert_allclose(cpu.coefficients, gpu_backend.to_numpy(gpu.coefficients), rtol=2e-5, atol=2e-5)
