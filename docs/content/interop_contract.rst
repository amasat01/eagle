Interoperability contract
=========================

.. note::

   New here? :doc:`interop` walks through numpy, cupy, torch and NVIDIA
   Warp crossing the same launcher, worked in a notebook, before this page's
   contract and refusal rules. hawk's `planes and roles
   <https://amasat01.github.io/hawk/content/vocabulary.html>`__ vocabulary
   (``Mutable``, ``Scalar``, ``Vector``) is what the shapes below are shapes
   *of*.

Any array that speaks DLPack — NumPy, CuPy, PyTorch, NVIDIA Warp, and by the
same protocol JAX, TensorFlow and others — becomes a first-class buffer for
the family: zero-copy, with correct stream ordering and lifetime. The layer
is implemented once, in C++ headers (``aether/interop/Buffer.h`` for the
buffer record, ``plugin/interop.h`` for the CUDA-aware half), and bound in
Python as :func:`eagle.interop.import_buffer`. A C++ consumer includes the
headers directly; nothing in them needs eagle's Python package.

.. code-block:: python

   import numpy as np
   from eagle.interop import Requirements, import_buffer

   a = np.zeros(1024)
   view = import_buffer(a, require=Requirements(dtype="float64", contiguous=True))
   view.access, view.owner, view.producer   # ('read-write', 'external', 'numpy.ndarray')
   view.ptr == a.ctypes.data                # True: zero-copy

Buffer model
------------

Every buffer sits on two independent axes:

- **ownership** — ``view.owner`` is the name of the allocating library when
  the memory belongs to the family (for example ``"eagle"``), and
  ``"external"`` when it is borrowed. ``view.producer`` is the producer's
  type name (``"numpy.ndarray"``, ``"cupy.ndarray"``, ``"torch.Tensor"``,
  ``"warp.array"``, or a C++ producer tag), ``None`` when owned. External
  memory is never freed by the consumer and never written beyond its access.
- **access** — ``view.access`` is reported exactly as the producer declared
  it:

  ============================================  ================
  producer form                                 ``view.access``
  ============================================  ================
  versioned DLPack, read-only flag set          ``"read-only"``
  versioned DLPack, flag clear                  ``"read-write"``
  legacy (pre-1.0) DLPack, no flag              ``"unknown"``
  array interface, ``data[1]`` true             ``"read-only"``
  array interface, ``data[1]`` false            ``"read-write"``
  ============================================  ================

  The layer never upgrades ``"unknown"`` by itself. A caller may assert
  writability with ``assume_writable=True``; the assertion is recorded
  (``view.assumed_writable``) and never overrides a declared read-only.

Policy belongs to the consumer, not the layer. A read-only kernel input
(``Table`` at user level, the ``lookup`` wire role) accepts any access; a
writable one (``Mutable``) requires ``"read-write"``, or ``"unknown"`` plus an
explicit assertion — ``Requirements(writable=True)`` is that check.

C++: ``enum class Access { ReadWrite, ReadOnly, Unknown }`` and
``aether::interop::BufferView`` carry the same fields; ``accessName`` gives
the shared spelling.

Import
------

``import_buffer(obj, *, stream=None, require=None, assume_writable=False)``
prefers ``obj.__dlpack__(stream=stream, max_version=(1, 0))``, falls back to
legacy ``__dlpack__``, then to ``__cuda_array_interface__`` (whose ``stream``
entry is fenced before ``stream``) and ``__array_interface__``. The view keeps
``obj`` alive: ``del obj`` afterwards is safe. It exposes ``access``,
``owner``, ``producer``, ``stream``, ``ptr``, ``shape``, ``strides`` (in
elements), ``dtype`` and ``device`` (``(device_type, device_id)`` in DLPack's
taxonomy), and re-exports through ``__dlpack__`` / ``__dlpack_device__``.

Stream codes
------------

``stream`` follows DLPack 1.0's CUDA semantics on both sides — the stream the
caller will use the buffer on at import, the consumer's stream at export:

=================  ====================================================
``stream``         meaning
=================  ====================================================
``None`` or ``1``  the legacy default stream
``2``              the per-thread default stream
``-1``             no ordering: the caller takes responsibility
``0``              refused (ambiguous by the DLPack specification)
any other int      a ``cudaStream_t`` handle
=================  ====================================================

``view.stream`` is the stream the view's contents are ordered on (an int),
``None`` for a host buffer. On export the layer records an event on that
stream and makes the consumer's stream wait on it — one
``cudaEventRecord`` plus one ``cudaStreamWaitEvent``, never a host
synchronization. Events come from a small per-device pool and are reused.

A producer that keeps writing between launches needs the ordering again:
call ``view.refence(stream)`` (or :func:`eagle.interop.refence`) before each
launch that reads the buffer. A host buffer takes no stream: any stream
argument other than ``None`` is refused.

``GraphPipeline.stream`` exposes a pipeline's own stream under the same
name, so a capture can import a buffer straight onto it.

Lifetime
--------

The view holds the producer. Each export stores a share of the original
keep-alive inside the exported tensor, so a chain import -> export -> import
-> ... keeps the FIRST producer alive until the last consumer anywhere in the
chain has released it; the producer's deleter runs exactly once. DLPack's
deleter is a plain host call: device work queued against the memory must be
ordered before the last reference goes away.

Refusals
--------

:class:`eagle.interop.Requirements` checks dtype, element count or shape,
C-contiguity with a unit innermost stride, alignment, device type and
ordinal, and writability. A failure raises
:class:`eagle.interop.BufferRefused` (a ``ValueError``) carrying one clause per
failed check, ``"<check>: required <want>, got <have>"``:

.. code-block:: text

   dtype: required float64, got float32
   count: required 10 elements, got 12
   stride: required C-contiguous with unit stride, got strides (2,) for shape (5,)
   device: required cuda, got cpu
   access: required read-write, got read-only

Refused stream codes raise ``ValueError`` naming the stream
(``stream: 0 is ambiguous ...``, ``stream: a CPU buffer takes no stream, got 1``).

Certification
-------------

NVIDIA Warp is certified today, through this same layer, not a dedicated
adapter: any ``__dlpack__`` + ``__dlpack_device__`` producer already reaches
:func:`eagle.interop.import_buffer`. A ``wp.array`` imports zero-copy
(``WP-IN-CUDA-ALIAS``) — Warp speaks legacy (pre-1.0) DLPack, so its access
flag always reports ``"unknown"`` — and exports back zero-copy via
``wp.from_dlpack`` (``WP-OUT-CUDA-ALIAS``); the stream contract above is
honoured end to end, producer side (``STREAM-WARP-PRODUCER-ORDER``). See
`raptor's interoperability protocols
<https://amasat01.github.io/raptor/content/interop_protocols.html>`_ for the
full, row-cited statement.

Layouts
-------

A per-sample vector plane is **component-major**: a ``Vector[3]`` plane over
``N`` samples is shaped ``(3, N)``, one contiguous row per component, so a
per-component access is coalesced. Most NumPy, PyTorch and JAX code stores
per-sample vectors **sample-major**, ``(N, 3)``. Every eagle door that binds a
per-sample plane (``Plan.run``, ``Plan.bind``/``BoundPlan.rebind``, the
``Loaded*`` kernels, and the torch bridge on host and device) accepts both:

- An array whose shape is the declared component-major one is bound as given.
- A square ``(3, 3)`` plane at ``N == 3`` reads both ways, as component-major
  ``(3, N)`` and as sample-major ``(N, 3)``, so eagle will not guess: it is
  refused, naming the argument, its shape, both readings and the two fixes
  below. A square matrix head with ``N == R == C`` is refused the same way.
  Both fixes are zero-copy: pass ``layout="samples_first"`` (planes are
  ``(N, 3)``) or ``layout="samples_last"`` (planes are ``(3, N)``) to the call
  (``Plan.run``, ``Plan.bind``, ``BoundPlan.rebind``, ``eagle.until_done``,
  ``eagle.run_until_done``, ``eagle.simulate``/``eagle.simulation``, the
  ``Loaded*`` kernels and the torch bridge) to resolve every ambiguous plane of
  that call, or wrap one argument: ``eagle.samples_first(x)`` /
  ``eagle.samples_last(x)``. The marker is honoured for any shape, a marker that
  contradicts the shape is refused naming the argument, and it wins over the
  call's ``layout=``. hawk's ``hawk.samples_first`` / ``hawk.samples_last`` are
  accepted too: a marker is any object carrying ``__raptor_samples_axis__``
  (``"first"`` or ``"last"``) and the wrapped array as ``.array``.
- A sample-major array whose transpose is C-contiguous (for example ``x.T`` of
  a contiguous ``(3, N)`` array) already holds the component-major bytes. It is
  bound as that view: no copy, no warning, input or output.
- Any other sample-major array is copied into a component-major buffer, and an
  ``eagle.LayoutWarning`` names the argument, both shapes, that zero-copy is
  lost for it, and how to avoid the copy. A written plane (a ``Mutable`` or an
  output) is copied back into the caller's array after the launch; a read-only
  input is copied in only. ``Plan.bind`` makes the copy once, at bind:
  ``BoundPlan.launch`` refreshes it from the caller's array before every launch
  and copies a written plane back after, on the launch's own stream, with no
  device-wide synchronization, so a captured graph replays the copies too.

The sample count is read from a sample-major plane's leading axis. Scalar
planes, lookup and scatter buffers and ``Param`` values are never adapted. A
matrix plane follows the same rules, with ``(N, R, C)`` or ``(N, R*C)`` as its
sample-major forms where the door accepts ``(R, C, N)``.

What each door hands back: ``Plan.run`` returns a caller-supplied output plane
in the caller's own layout (the array itself, for a host array) — and also
WRITES the result into that same array in place, host or device structure,
before returning it; a read-only supplied array is refused by name rather
than silently left untouched. The torch
bridge returns its outputs in the caller's own layout too — sample-major only
when every per-sample input was, as a zero-copy transposed view, and refused
when the inputs disagree — and the gradient of a sample-major input comes
back sample-major. A ``Loaded*`` kernel returns its updated ``Mutable``
buffers component-major and also writes a copied one back into the caller's
array.

``eagle.LayoutWarning`` is a ``UserWarning`` subclass, issued once per call
site and argument. To make every copy an error instead, raised before anything
is copied:

.. code-block:: python

   import warnings

   import eagle

   warnings.filterwarnings("error", category=eagle.LayoutWarning)

hawk's own host runtime (``hawk.runtime``) binds by address and never copies:
it accepts the zero-copy view and refuses every other sample-major plane with
the same fix named.

One sample
~~~~~~~~~~

One sample runs through the same kernel as a batch. A value whose shape IS a
plane's per-sample head is one sample: a Python number or a 0-d array for a
``Scalar``/``Index`` plane, ``(w,)`` for a ``Vector[w]``/``Quat``, ``(R, C)``
or the flat ``(R*C,)`` for a ``Matrix[R, C]``, and a 0-d ``bool`` array for a
``Terminated`` mask. ``Plan.run``, ``Plan.bind``, ``eagle.until_done`` and
``eagle.run_until_done`` decide this before reading the sample count, on host
and device alike:

1. ``head + (N,)`` is a native batch and ``(N,) + head`` a sample-major one,
   as above.
2. A shape equal to the head is ONE sample.
3. A shape that is both a batch and a sample is a batch, never guessed away:
   ``(1,)`` on a scalar plane is a batch of one (a 1-D array on a scalar plane
   is always a batch), ``(w, 1)`` is a batch of one, ``(w, w)`` a batch of
   ``w`` (ambiguous: refused until ``layout=`` or a marker says which axis
   holds the samples), and a ``Matrix[R, 1]`` given ``(R, 1)`` a batch of one.
4. Lookup, wide and accumulate buffers keep their own lengths, a ``Reduce``
   output stays ``(L,)``, and a ``Param`` is already one value for every
   sample.
5. A call is one sample or one batch, never both. A call that mixes them is
   refused, naming both: ``'x' is one sample (shape (3,)) but 's' is a batch
   of 1000 (shape (1000,))``, with the fix: a value shared by every sample is a
   ``Param`` (scalar) or a ``Table`` (vector), or tile it once.
6. One sample runs as a batch of one, through the zero-copy view
   ``value.reshape(head + (1,))``.

``Plan.run`` returns each output in its head shape, in the caller's framework:
a ``numpy.float64`` (a ``float`` with ``.shape == ()``) for a scalar, ``(w,)``
and ``(R, C)`` arrays otherwise, and cupy arrays (0-d for a scalar) when the
inputs are on the device. A batch of one (``(1,)``, ``(w, 1)``) still returns
batch shapes. A Python number is accepted for an input only: an output given a
number is refused (``output 'y' was given a Python float, which cannot receive
a write``), and a supplied head-shaped output (``np.zeros(())``,
``np.zeros(w)``) is written in place and returned itself.

``Plan.bind``, ``eagle.until_done`` and ``eagle.run_until_done`` allocate
nothing, so one sample is bound as head-shaped ARRAYS, written in place; the
``terminated`` mask is a 0-d ``bool`` array or omitted. A number for a
``Mutable`` is refused with the fix ``np.array(x)``.

Where one sample runs is the data's choice, never the sample count's:
``eagle.plan.auto(plugin)`` returns an ``AutoPlan`` whose ``run``/``bind``
select the host plan when every per-sample value is host-resident (numpy, a
number) and the device plan when any is device-resident (cupy); data on both
sides is refused (``move one side with cp.asarray / .get()``), and so is a
side the plugin was not built for. ``eagle.until_done`` accepts an
``AutoPlan``. ``eagle.deploy(kernel)`` (the same function as
``eagle.plan.auto``) takes a hawk kernel itself, and ``eagle.deploy([k1, k2])``
a list (one bundle, a tuple of plans in order): it builds through hawk's
cache and assembles the plugin with ``hawk.artifact.plugins``. hawk's host runtime restates rules 1 to 3 and 5
(``hawk.runtime.plane_layout`` answers ``"single"``) and binds one sample as a
0-d or ``(w,)`` array; it refuses a Python number, since it binds by address.

Roadmap
-------

DLPack is spoken by many more frameworks than are certified here. JAX,
TensorFlow and Keras reach this layer by the same protocol, but only the
rows of raptor's interoperability certification matrix are claimed as
supported; this page makes no claim beyond them.

API
---

.. autofunction:: eagle.interop.import_buffer

.. autoclass:: eagle.interop.BufferView
   :members:

.. autoclass:: eagle.interop.Requirements

.. autoclass:: eagle.interop.BufferRefused

.. autofunction:: eagle.interop.refence
