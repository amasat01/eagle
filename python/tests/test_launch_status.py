# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""No silent failed launch: every device launch eagle issues reads the launch status afterwards (a query, no wait) and
raises a failure by name, eager and graph replay alike; a bind whose kernel keeps more local memory than the device can
hold across its resident threads is refused by name."""

import pytest

hawk = pytest.importorskip("hawk")
import eagle  # noqa: E402
import eagle.exec as eexec  # noqa: E402
import eagle.plan  # noqa: E402
from eagle import _split  # noqa: E402
from eagle.pipeline import GraphPipeline  # noqa: E402
from hawk import Mutable, Scalar  # noqa: E402
from hawk.artifact import build_bundle, plugins  # noqa: E402

pytestmark = pytest.mark.gpu


@hawk.kernel
def square(x: Scalar, y: Mutable[Scalar]):
    y = x * x


@pytest.fixture(scope="module")
def plan_(tmp_path_factory):
    cp = pytest.importorskip("cupy")  # noqa: F841
    root = tmp_path_factory.mktemp("status")
    bundle = build_bundle([square], root / "b", targets=("cuda",), cache_dir=str(root / "c"), host_profile="native")
    by = plugins(bundle, device_loader=eagle.registry.load_manifest)
    _split.link(by, bundle)
    return eagle.plan.plan(by["square"], structure=eexec.DeviceKernel)


@pytest.fixture
def inject(monkeypatch):
    """``inject()`` makes the next launch-status read report an error, as the plugin's runtime does after a
    launch it issued fails (a failure of the driver's own launch call is raised on the spot, and is tested as
    such: the plugin's runtime state is not reachable from a test)."""
    from eagle import _core

    real = _core.launch_status

    def arm():
        def once(device=-1):
            monkeypatch.setattr(_core, "launch_status", real)
            raise RuntimeError("CUDA launch error cudaErrorLaunchOutOfResources: injected")
        monkeypatch.setattr(_core, "launch_status", once)

    return arm


def test_an_eager_launch_that_leaves_an_error_raises_by_kernel_name(plan_, inject):
    import cupy as cp

    bound = plan_.bind(x=cp.ones(100), y=cp.zeros(100))
    bound.launch()
    inject()
    with pytest.raises(RuntimeError, match=r"kernel 'square': the launch failed \(CUDA launch error"):
        bound.launch()
    bound.launch()                                      # read once: the next launch is clean


def test_a_launch_the_driver_refuses_is_named(plan_):
    import cupy as cp

    bound = plan_.bind(x=cp.ones(100), y=cp.zeros(100))
    with pytest.raises(RuntimeError, match=r"kernel 'square': the launch failed"):
        bound.launch(block=4096)                        # more threads per block than any device allows


def test_a_graph_replay_and_its_instantiate_raise_by_name(plan_, inject):
    import cupy as cp

    bound = plan_.bind(x=cp.ones(100), y=cp.zeros(100))
    pipe = GraphPipeline().add(bound.launch).build()
    pipe.launch()
    inject()
    with pytest.raises(RuntimeError, match=r"kernel 'graph \[square\] at replay': the launch failed \(CUDA launch error"):
        pipe.enqueue()
    pipe.synchronize()
    inject()
    with pytest.raises(RuntimeError, match=r"the launch failed \(CUDA launch error"):
        GraphPipeline().add(bound.launch).build()


def test_plan_run_names_a_failed_launch(plan_, inject):
    import cupy as cp

    inject()
    with pytest.raises(RuntimeError, match=r"kernel 'square': the launch failed"):
        plan_.run(x=cp.ones(100))


def test_a_kernel_whose_local_memory_will_not_fit_is_refused_at_bind(plan_, monkeypatch):
    import cupy as cp

    planes = {"x": cp.ones(100000), "y": cp.zeros(100000)}
    plugin = plan_.plugin
    for field, value in (("local_bytes", 41_000), ("local_source", "ptxas"), ("local_limit", 524_288)):
        monkeypatch.setattr(plugin, field, value, raising=False)
    monkeypatch.setattr(_split, "_free_bytes", lambda device: 1_000_000)
    with pytest.raises(ValueError, match=r"kernel 'square' keeps 41,000 bytes of local memory per thread \(ptxas\) over "
                                         r"[\d,]+ resident threads = [\d,]+ bytes, but the device has 1,000,000 bytes free"):
        plan_.bind(**planes)
    monkeypatch.setattr(plugin, "local_bytes", 600_000, raising=False)
    with pytest.raises(ValueError, match=r"needs 600,000 bytes of local memory per thread, past the device limit"):
        plan_.bind(**planes)
    monkeypatch.setattr(plugin, "local_bytes", None, raising=False)       # an older sidecar: not checked
    plan_.bind(**planes)


def test_an_unlinked_plugin_with_over_limit_bytes_is_refused_at_bind(tmp_path):
    """``plugins()`` carries the launch block's local memory fields itself; no ``link`` is needed."""
    import cupy as cp

    root = tmp_path
    bundle = build_bundle([square], root / "b", targets=("cuda",), cache_dir=str(root / "c"), host_profile="native")
    plugin = plugins(bundle, device_loader=eagle.registry.load_manifest)["square"]
    if getattr(plugin, "local_bytes", None) is None:
        pytest.skip("this hawk's plugin projection carries no launch.local_bytes")
    plugin.local_bytes, plugin.local_limit = 600_000, 524_288
    with pytest.raises(ValueError, match=r"kernel 'square' needs 600,000 bytes of local memory per thread"):
        eagle.plan.plan(plugin, structure=eexec.DeviceKernel).bind(x=cp.ones(10), y=cp.zeros(10))


def test_a_host_node_graph_runs_clean_and_a_not_permitted_error_still_raises(plan_, monkeypatch):
    """A graph's host node reads no launch status (no CUDA call is legal in its callback); a normal launch whose status
    read answers "operation not permitted" raises by name."""
    import cupy as cp
    from eagle import _core, _stream

    seen = []

    def watch():
        _stream.check_launch("inside the callback")          # a host node's callback: must not call the backend
        seen.append(1)

    x, y = cp.ones(100), cp.zeros(100)
    g = eagle.Graph().add(plan_.bind(x=x, y=y), name="sq").add_node(eagle.opaque(watch, reads=(y,), writes=(), eager=True)).build()
    g.launch()
    assert seen
    real = _core.launch_status

    def refuse(device=-1):
        raise RuntimeError("eagle: cudaGetDevice: operation not permitted")

    monkeypatch.setattr(_core, "launch_status", refuse)
    with pytest.raises(RuntimeError, match=r"kernel 'square': the launch failed \(.*operation not permitted"):
        plan_.bind(x=x, y=y).launch()
