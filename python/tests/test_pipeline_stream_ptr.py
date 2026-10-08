# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``GraphPipeline.stream`` -- the public read-only accessor for the
pipeline's own ``cudaStream_t`` (the family's shared ``stream`` name;
previously unpublished, which forced external capture composition to reach
for the private ``_sptr`` instead -- cuBLAS handle binding + warm passes need
to run on the pipeline's own stream).

``GraphPipeline.__init__`` creates its ``cudaStream_t`` through the bound
``eagle::cuda`` core, which needs a real CUDA context -- there is no
host-only construction path, so this whole file is ``gpu``-marked (mirrors
``test_pipeline_parity.py``/``test_pipeline_concurrent.py``/
``test_graph_plugin.py``'s own module-level ``pytestmark``).
"""

import pytest

from eagle.pipeline import GraphPipeline

pytestmark = pytest.mark.gpu


def test_stream_property_exists_and_is_read_only():
    pipe = GraphPipeline()
    assert isinstance(pipe.stream, int)
    with pytest.raises(AttributeError):
        pipe.stream = 0


def test_stream_matches_the_external_stream_cupy_was_handed():
    """``self._ext`` (the ``cp.cuda.ExternalStream`` cupy launches on) is
    built from the exact same raw handle ``stream`` now publishes --
    the property must not be a second, independently-derived value."""
    pipe = GraphPipeline()
    assert pipe.stream == pipe._ext.ptr
