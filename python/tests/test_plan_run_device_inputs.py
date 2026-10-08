# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Plan.run`` on a DEVICE plan takes device-resident inputs where they lie.

Before: ``_host_plane`` called ``np.asarray`` on every input, so a cupy input
was refused by cupy itself (implicit device-to-host conversion is not allowed)
and a device caller had to bring its data home only for the plan to upload it
again. Now a cupy (or CUDA torch) input is bound in device memory and, when any
input is device-resident, the outputs stay on the device as cupy arrays.

The non-vacuity half of every row: ``cupy.asnumpy`` is patched to raise for the
whole call, the returned planes must be cupy arrays, and a supplied output must
come back at the caller's own device pointer — a host round trip anywhere in
the path fails the row.
"""

from __future__ import annotations

import numpy as np
import pytest

import eagle

from test_plan_packer_e2 import (  # noqa: E402
    _VEC3_SPEC,
    _device_plugin,
    _x,
    e2_device_module,  # noqa: F401 -- reused as a pytest fixture, by parameter name
)
from test_plan_run_out import (  # noqa: E402
    _SCALE_SPEC,
    scale_device_module,  # noqa: F401 -- reused as a pytest fixture, by parameter name
)

pytestmark = [pytest.mark.repo_local, pytest.mark.gpu]


@pytest.fixture
def no_download(monkeypatch):
    """Refuse every ``cupy.asnumpy`` for the duration of the row."""
    import cupy as cp

    def refuse(*_a, **_kw):
        raise AssertionError("Plan.run brought a device plane home (cupy.asnumpy)")

    monkeypatch.setattr(cp, "asnumpy", refuse)
    return cp


def _scale_plan(module):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_plugin(module, "scale", _SCALE_SPEC, arg_widths={"y": 1})
    return eplan.plan(plugin, structure=eexec.DeviceKernel)


def _vec3_plan(module):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_plugin(module, "e2_vec3", _VEC3_SPEC, arg_widths={"y": 3})
    return eplan.plan(plugin, structure=eexec.DeviceKernel)


def test_cupy_input_returns_a_device_output(scale_device_module, no_download):
    cp = no_download
    x = cp.arange(8, dtype=cp.float64)
    y = _scale_plan(scale_device_module).run(x=x, a=2.0, b=1.0)
    assert isinstance(y, cp.ndarray)
    np.testing.assert_array_equal(y.get(), 2.0 * np.arange(8.0) + 1.0)


def test_cupy_input_writes_the_supplied_device_output_in_place(
    scale_device_module, no_download
):
    cp = no_download
    x = cp.arange(8, dtype=cp.float64)
    y = cp.zeros(8)
    got = _scale_plan(scale_device_module).run(x=x, a=-1.0, b=0.5, y=y)
    assert isinstance(got, cp.ndarray)
    assert got.data.ptr == y.data.ptr                # the caller's own buffer
    np.testing.assert_array_equal(y.get(), -np.arange(8.0) + 0.5)


def test_cupy_input_of_another_dtype_is_cast_on_the_device(
    scale_device_module, no_download
):
    cp = no_download
    x = cp.arange(8, dtype=cp.float32)[::-1]          # float32 and non-contiguous
    y = _scale_plan(scale_device_module).run(x=x, a=1.0, b=0.0)
    assert isinstance(y, cp.ndarray) and y.dtype == cp.float64
    np.testing.assert_array_equal(y.get(), np.arange(8.0)[::-1])


def test_sample_major_device_output_is_written_back_with_a_warning(
    e2_device_module, no_download  # noqa: F811
):
    cp = no_download
    n = 16
    x = cp.asarray(_x(n))
    y = cp.zeros((n, 3))                              # sample-major, needs a copy
    with pytest.warns(eagle.LayoutWarning, match=r"'y'.*copied back"):
        got = _vec3_plan(e2_device_module).run(x=x, a=2.0, y=y)
    assert got is y
    expected = 2.0 * _x(n) + np.arange(3, dtype=np.float64)[:, None]
    np.testing.assert_array_equal(y.get(), expected.T)


def test_transposed_device_view_binds_zero_copy(e2_device_module, no_download):  # noqa: F811
    import warnings

    cp = no_download
    n = 16
    x = cp.asarray(_x(n))
    y_native = cp.zeros((3, n))
    with warnings.catch_warnings():
        warnings.simplefilter("error", eagle.LayoutWarning)
        got = _vec3_plan(e2_device_module).run(x=x, a=2.0, y=y_native.T)
    assert got.data.ptr == y_native.data.ptr
    expected = 2.0 * _x(n) + np.arange(3, dtype=np.float64)[:, None]
    np.testing.assert_array_equal(y_native.get(), expected)


def test_torch_cuda_input_returns_a_device_output(scale_device_module, no_download):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA in this environment")
    cp = no_download
    x = torch.arange(8, dtype=torch.float64, device="cuda")
    y = torch.zeros(8, dtype=torch.float64, device="cuda")
    got = _scale_plan(scale_device_module).run(x=x, a=3.0, b=0.0, y=y)
    assert isinstance(got, cp.ndarray)
    assert got.data.ptr == y.data_ptr()
    np.testing.assert_array_equal(y.cpu().numpy(), 3.0 * np.arange(8.0))


def test_host_inputs_still_return_host_outputs(scale_device_module):
    import cupy as cp

    y_dev = cp.zeros(8)
    got = _scale_plan(scale_device_module).run(x=np.arange(8.0), a=2.0, b=0.0, y=y_dev)
    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, 2.0 * np.arange(8.0))
    np.testing.assert_array_equal(y_dev.get(), 2.0 * np.arange(8.0))
