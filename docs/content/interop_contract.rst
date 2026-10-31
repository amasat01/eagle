Interoperability contract
=========================

.. note::

   New here? :doc:`interop` walks through numpy, cupy, torch, NVIDIA
   Warp and JAX crossing the same launcher, worked in a notebook, before this page's
   contract and refusal rules. hawk's `planes and roles
   <https://amasat01.github.io/hawk/content/vocabulary.html>`__ vocabulary
   (``Mutable``, ``Scalar``, ``Vector``) is what the shapes below are shapes
   *of*.

Any array that speaks DLPack — NumPy, CuPy, PyTorch, NVIDIA Warp or JAX —
becomes a first-class buffer for the family: zero-copy, with correct stream ordering and lifetime. The layer
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

Your stream, and results in your library
----------------------------------------

**eagle's stream.** With nothing open, eagle works on the default stream. It never reads CuPy's or torch's
current stream, so opening one of theirs changes nothing for eagle. To put eagle's work on your stream, for one
call pass ``stream=``, or for a block write ``with eagle.stream(s):``. ``s`` is a raw handle, an object with
``__cuda_stream__``, a torch or Warp stream (``.cuda_stream``) or a CuPy stream (``.ptr``); eagle turns it into a
raw handle once, at that boundary. Inside your own CuPy or torch capture, ``with eagle.stream(s):`` puts eagle's
launches in the same capture.

A raw int handle is your contract. eagle does not check that it is a live stream, or that it belongs to the right
device. Pass a stream object when you have one.

**Results.** eagle returns ``eagle.Array``: eagle's own array over aether memory, which every library above
reads through DLPack or ``__cuda_array_interface__`` and which speaks the buffer protocol on the host. To get a
library's own type, ask for it:

.. code-block:: python

   y = plan.run(x=x, ...)                         # eagle.Array
   y_np = y.to("numpy")                           # "numpy", "cupy", "torch", "jax" or "warp"
   y_t = plan.run(x=x, ..., returns="torch")      # every result, converted

``to()`` shares the memory where the library can (DLPack with your stream handed over) and copies otherwise. If the
library is not installed, ``to()`` raises ``ImportError`` with the install line, and ``returns=`` warns once and
returns ``eagle.Array``. ``eagle.to_array(x)`` goes the other way, zero-copy. A complex array is refused by
name: hold the ``(re, im)`` pairs as a float array with a trailing axis of 2.

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

NVIDIA Warp is certified through this same layer: any ``__dlpack__`` + ``__dlpack_device__`` producer already reaches
:func:`eagle.interop.import_buffer`. A ``wp.array`` imports zero-copy
(``WP-IN-CUDA-ALIAS``) — Warp speaks legacy (pre-1.0) DLPack in the version the
tests pin, so its access flag reports ``"unknown"`` — and exports back zero-copy via
``wp.from_dlpack`` (``WP-OUT-CUDA-ALIAS``); the stream contract above is
honoured end to end, producer side (``STREAM-WARP-PRODUCER-ORDER``). See
`raptor's interoperability protocols
<https://amasat01.github.io/raptor/content/interop_protocols.html>`_ for the
full, row-cited statement.

``eagle.frameworks.warp`` runs a hawk kernel on Warp arrays directly, with ``wp.Tape`` gradients and
CUDA graph capture; the Warp page walks through it.

JAX is certified the same way. JAX publishes wheels for Python 3.12 and
newer, and no 3.13 free-threaded build; its CUDA 12 build supports Pascal
and newer, and its CUDA 13 build supports newer GPUs only. JAX arrays are
immutable, so the rules differ from the writable frameworks:

* A JAX array in a read-only role (state vectors, per-sample planes, tables,
  uniforms) is imported without a copy on CUDA (``JX-IN-CUDA-ALIAS``) and
  uploaded on the CPU (``JX-IN-CPU-COPY``). JAX speaks legacy DLPack, so its
  access flag reports ``"unknown"``.
* A JAX array in a role the kernel writes (``out=``, ``terminated``, a
  ``Mutable`` plane, a written host buffer) is refused by name and left
  unchanged (``JX-IN-WRITE-REFUSED``), because a copy would drop the kernel's
  update. A plugin that declares ``terminated`` read-only accepts one.
* The JAX result on CUDA is eagle's own result buffer; eagle never writes it
  again (``JX-OUT-CUDA-ALIAS``). Zero-copy in, zero-copy out, never written in
  place.
* eagle imports a JAX array with its current stream passed to ``__dlpack__``,
  so a JAX operation still running on the GPU is waited for on the device
  (``STREAM-JAX-PRODUCER-ORDER``). The result is synchronized before it is
  handed back to JAX.

The bridge ``eagle.frameworks.jax.function``, which runs a hawk kernel as an
XLA custom call on XLA's own buffers and stream, is certified by four more rows:

* On the CPU platform the kernel reads XLA's host buffers in place and the
  result is a fresh XLA-owned buffer (``JX-FFI-CPU-ALIAS``).
* Outputs are XLA-allocated and zero where the kernel writes nothing, on CPU and
  CUDA, even when the allocator held old data; a donated mask comes back at the
  same address and one that was not donated is left intact
  (``JX-OUT-XLA-OWNED``).
* Eager, ``jit``, a ``vmap``-folded batch and a ``lax.scan`` of steps agree: bit
  for bit on CUDA and on the exact host profiles, within a few ULP on
  ``native-vector-math`` (``JX-JIT-INVARIANCE``).
* In one ``jit``, a kernel after a long-running XLA operation runs on XLA's
  stream: the call returns without a host wait and reads the finished values
  (``STREAM-JAX-FFI-ORDER``).
* ``grad``, ``jit(grad)``, ``vmap(grad)``, ``jacfwd`` and ``jacrev`` of a kernel
  match the same law written in plain JAX, on CPU and CUDA
  (``JX-GRAD-TRANSFORMS``).

Three more rows cover JAX code inside an eagle graph:

* A JAX step between eagle stages runs in order on the device, its result stays
  alive until eagle's copy of it is done, and the host does not wait
  (``STREAM-JAX-INJECT-ORDER``).
* A JAX call in a host-synchronized node is seen by the next stage
  (``JX-INJECT-HOST``).
* A carried-state kernel in ``jit(while_loop)`` gives the run-until-done answer,
  and ``grad`` of a ``scan`` over it matches a plain JAX reference
  (``JX-LOOP-INVARIANCE``).

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
``AutoPlan``, and planes from any DLPack producer (JAX, torch, CuPy, NumPy): it reads them in place, runs
on the side the data is on (a host plane beside a device plane is refused by name), and never writes an
array of another framework (eagle's own tests cover this; no row of the certification matrix does yet). ``Runner.result(name)`` returns a plane the kernel writes as a new array of the
caller's framework; a NumPy or CuPy plane is written in place as before. ``eagle.deploy(kernel)`` (the same function as
``eagle.plan.auto``) takes a hawk kernel itself, and ``eagle.deploy([k1, k2])``
a list (one bundle, a tuple of plans in order): it builds through hawk's
cache and assembles the plugin with ``hawk.artifact.plugins``. hawk's host runtime restates rules 1 to 3 and 5
(``hawk.runtime.plane_layout`` answers ``"single"``) and binds one sample as a
0-d or ``(w,)`` array; it refuses a Python number, since it binds by address.

The lane count of a ``Grid`` kernel
-----------------------------------

A hawk kernel can declare the shape of its lanes, ``lane: Grid["channel", "sample":50]``
(see hawk's *Across lanes*). The plan reads that declaration from the kernel's sidecar
(``launch.lane_space``) when it binds, so the caller gives the lane count in the kernel's own
terms:

.. code-block:: python

   import numpy as np
   import eagle
   import hawk
   from hawk import Grid, Scalar


   @hawk.kernel
   def per_channel(lane: Grid["channel", "sample":50], x: Scalar) -> Scalar:
       return hawk.sum(x * x, over="sample")


   plan = eagle.deploy(per_channel)
   x = np.random.default_rng(1).normal(size=(4, 50))
   print(plan.run(x=x).shape, plan.run(x=x, channel=4).shape)

.. code-block:: text

   (4,) (4,)

- The runtime extent is the first axis's label (``channel=4``), or comes from a lane-shaped
  plane: ``x`` of shape ``(4, 50)``, or ``(w, 4, 50)`` for a vector plane. The plane is used as
  a flat view; nothing is copied.
- ``n=`` is refused on such a kernel and names the label to pass instead. A plane whose
  static axes differ from the declaration is refused by name.
- Outputs come back lane-shaped. A reduced output (``hawk.sum(..., over=)``) or a
  ``Mutable["bin":256]`` sink is allocated in its own shape and zeroed; one you pass must match
  that shape, and keeps its contents.
- A launch that binds only ``Table`` planes has no lane count to read, since a table's length is
  never the lane count; it is refused, for every kernel. Pass a per-sample plane or the extent.
- An ``Index`` plane has the width the kernel declares (``int32`` unless declared ``Index[int64]``). ``bind``
  and ``rebind`` refuse another width by name and say which type to pass; ``run`` converts per call and
  ``eagle.simulate`` once at entry, so nothing is copied at a launch.
- ``Plan.rebind`` and ``eagle.until_done`` take lane-shaped planes the same way. A kernel
  with a ``group`` compacts with ``ActiveSet(mask, group=G)``: see
  :doc:`modules/filtering/scan_and_compaction`.

The block size of a kernel
--------------------------

A kernel whose sidecar states ``launch.block`` was emitted for exactly that many threads per
block (hawk writes it when an emit fixes one). Every device launch of its plan uses that
size, and the launch policy does not choose another. A ``block=`` argument that agrees is
accepted; one that contradicts it is refused, naming both:

.. code-block:: text

   ValueError: eagle.launch: block=128 contradicts the artifact's launch.block 256
   (the kernel was emitted for exactly that many threads per block)

A kernel that states no block keeps the policy's choice, and ``block=`` pins it as before. A
``launch.block`` that is not an integer of at least 1 is refused when the plan loads, naming the key.

Roadmap
-------

DLPack is spoken by many more frameworks than are certified here.
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
