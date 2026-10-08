The parallel-execution model
=============================

.. contents:: On this page
   :local:
   :depth: 1

Read this before :doc:`foundations` if terms like "kernel," "thread," or
"device memory" aren't already familiar. It explains the one idea
everything else in these docs assumes: *you write what happens to a single
item, and the hardware runs that same logic across every item at once.*
Nothing here is EAGLE-specific — it's the same idea whether the "hardware"
running your logic is a GPU or a multi-core CPU, which is exactly why
EAGLE's API has one shape for both.

One body, run many times at once
----------------------------------

Ordinary sequential code processes items one at a time, in a loop:

.. code-block:: cpp

   for (int i = 0; i < N; i++) {
       result[i] = a[i] + b[i];   // one item, then the next, then the next...
   }

On a single CPU core, this genuinely happens one iteration after another.
But the *body* of that loop — ``result[i] = a[i] + b[i]`` — doesn't care
what order the iterations happen in, or whether they happen at the same
time. That's the property both GPUs and modern CPUs exploit: instead of
running the body ``N`` times in sequence, run it for many values of ``i``
*simultaneously*, on separate pieces of hardware that all execute the exact
same instructions at once. A **kernel** (the term used throughout these
docs, and throughout GPU programming generally) is exactly that: the body
of a loop, written once, launched to run across every item at once instead
of one at a time.

What actually runs it, in parallel
------------------------------------

- **On a GPU**, each iteration runs on its own **thread** — a GPU has
  thousands of small, simple compute units, and a single kernel launch puts
  one of them on each item (or a small group of items) so they all execute
  the same instructions in lockstep.
- **On a CPU**, you have far fewer cores, but each core's vector hardware
  (SIMD) can run the *same* instruction over several values at once. EAGLE
  calls one such group of items processed together a **packet** (of a fixed
  size ``W``, the CPU's SIMD width), and each item's slot within that packet
  a **lane** — a packet is the CPU's version of a GPU thread group. See
  :doc:`foundations` for the full walkthrough of this vocabulary in EAGLE's
  ``cpu::Host`` dispatcher.

Both are the same underlying trick — do the same operation to many items at
once instead of one at a time — implemented on different hardware. That's
why EAGLE presents one calling shape (write the per-item body, call
``launch``) for both: the *parallel-execution idea* is identical; only the
hardware executing it differs, and EAGLE's dual-mode API is what lets your
code not care which one it's running on.

Host and device memory are separate
--------------------------------------

A CPU (the **host**) and a GPU (the **device**) each have their own,
physically separate memory. Data your CPU code has doesn't automatically
exist where the GPU can read it — it has to be explicitly copied over
(**uploaded**) before a GPU kernel can use it, and copied back
(**downloaded**) to read the result on the CPU side. This is why EAGLE's
APIs have explicit ``upload()`` / ``download()`` steps (see
:doc:`../modules/module_filtering` for a concrete example) — it's not
EAGLE-specific bookkeeping, it's a direct consequence of the GPU having its
own memory.

What you won't find explained here
--------------------------------------

Raw CUDA's own kernel-launch syntax — the ``<<<grid, block, shmem,
stream>>>`` triple-angle-bracket notation you'll see in
:doc:`quickstart`'s ``addOne<<<1, N, 0, stream.cuda()>>>(...)`` — along with
what a "block" and "grid" precisely are, and the full CUDA memory model, are
raw CUDA C++ concepts that EAGLE builds on top of but doesn't reinvent.
EAGLE's own API (``cpu::Host::launch``, ``GraphPipeline``, the ``Reduction``
/ ``Scanner`` helpers) is designed so you rarely need to write that syntax
yourself. If you do want the full picture — writing a raw CUDA kernel from
scratch, tuning block/grid sizes, the GPU memory hierarchy in depth —
`NVIDIA's own CUDA C++ Programming Guide
<https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>`_ is the
authoritative reference; nothing in these docs assumes you've read it, but
it's the right place to go deeper.
