# Glossary

The words eagle's documentation uses, each in a sentence. Terms shared by the
whole RAPTOR family (manifest, sidecar, DLPack and so on) are defined once in
the [family glossary](https://amasat01.github.io/raptor/content/glossary.html).

```{glossary}
active set
: The samples still running; {class}`eagle.ActiveSet` keeps them as an index map instead of moving data.
adjoint
: The quantity flowing backwards through the graph: for a plane ``p``, ``∂J/∂p``. A seed is the adjoint of an output; a gradient is the adjoint of an input. Where two nodes read one plane, the plane's adjoint is the sum of what flows back from each.
bitwise
: Bit-for-bit reproducible: the same inputs give the same bits on every run and every thread count. Every gradient eagle sums is bitwise; the one exception is named in {ref}`autodiff-bits`.
canonical tree
: The one fixed grouping of additions eagle uses for sums, so a total has the same bits for any thread count.
cell
: One value for the whole batch, made by a hawk kernel that stores a reduction over samples (`hawk.all_samples(x).sum()`) to a `Mutable` output. The kernel writes rows; eagle finishes them into the cell with {func}`eagle.reduce`; see {ref}`determinism-cells`.
compaction
: Shrinking a per-sample array to just the samples that are still active, so later kernels launch over a smaller count; also called stream compaction.
dual-mode
: A source file that compiles both against CUDA and as plain C++ with OpenMP, with no change to the calling code.
event refinement
: Locating a stop *inside* the step where its rule fires (``events="refine"``), so the stop time is a real number with a derivative instead of a multiple of the step; see {ref}`autodiff-events`.
generation
: Which launch a Record belongs to: 1 for the first launch, 2 for the second, and so on. It is how a reverse knows its Record is still the newest one.
gradient
: What comes back: ``∂J/∂input``. For a per-sample input, one value per sample (in sample order); for a parameter shared by every sample, one value, summed over the samples.
graph
: A recorded, dependency-ordered set of launches (a CUDA graph on the GPU, a host executor on the CPU) that is built once and replayed many times.
graph capture
: Recording the kernel launches issued on a stream, instead of running them, into a graph that can be replayed.
guard
: The cell a conditional node reads: an {func}`eagle.skippable` runs its step only while the cell is non-zero. A recorded graph saves the value each launch saw, and the reverse replays it; see {ref}`autodiff-conditionals`.
lane
: One sample's slot inside a packet on the CPU side.
launch
: One submission of work: a single kernel call, or one replay of a whole graph.
Launcher (C++)
: ``eagle::cuda::Launcher``, the instantiated, replayable form of a graph; calling its ``launch()`` repeats the recorded work with no per-launch setup. Not the same as the family's *launcher* protocol (see the family glossary).
mask
: A per-packet record of which lanes are still active, so inactive lanes (or whole packets) can be skipped.
memory budget
: The bytes a recorded run's reverse may hold (``memory_budget=``). Over it, the reverse runs the batch in passes with the same bits; see {ref}`autodiff-budget`.
packet
: A fixed-size group of samples that the host launcher processes together with one SIMD instruction stream.
phase
: One continuous piece of a hybrid trajectory: a refined run made a node of a Graph by {func}`eagle.diff.phase`, with a reset kernel between pieces; see {ref}`autodiff-events`.
plan
: In eagle, a plan is what `eagle.deploy` returns: a kernel built to run where its data lives, ready to launch. The general meaning, shared by the family, is in the [family glossary](https://amasat01.github.io/raptor/content/glossary.html#term-plan).
Record
: What one launch of a ``record=True`` graph keeps for its reverse: ``g.record`` after ``g.launch()``. Most of it is the graph's own planes, so it is only valid until the graph runs again (see {ref}`autodiff-record`).
reduction policy
: How {func}`eagle.reduce` adds the leaf results of a floating-point sum: `"compensated"` (default, accurate, not bitwise on the GPU), `"ordered"` (the same bits every run) or `"auto"` (`"compensated"` for now).
registry
: The table that resolves a compiled kernel by name so it can be called, streamed or put in a graph ({class}`eagle.KernelRegistry`).
rows
: What a kernel that reduces over samples writes: one partial per 32 samples, plus a `_comp` row of rounding corrections for a sum. Finishing them gives the {term}`cell`.
reverse mode
: Computing derivatives by running the program's steps backwards, each step turning "how the objective moves with my outputs" into "how it moves with my inputs". One backward sweep gives the gradient with respect to *every* input at once, which is why it is the method of choice when you have one objective and many inputs. Also called the adjoint method (optimal control, PDE-constrained optimisation) or backpropagation (machine learning).
stage
: One launch of a kernel that reads a value reduced over samples: the read is a launch boundary, so eagle runs the stages in sequence with the finish of the cells between them; see {ref}`determinism-cells`.
sample
: One independent item the same kernel runs over: a particle, a trajectory, a network input. eagle runs every sample through the same step.
seed
: What you hand the reverse pass: the gradient of your objective ``J`` with respect to an output plane, ``∂J/∂output``, one value per sample. If ``J`` is the sum of the outputs, every seed is ``1``; if ``J`` is a weighted sum, the seed is the weights; if ``J`` is a terminal cost ``φ(x(T))``, the seed is ``∂φ/∂x(T)``. It is per sample because each sample has its own output, so each has its own ``∂J/∂output``. Also called a cotangent (JAX), ``grad_outputs`` (PyTorch), ``adj_seed`` (CasADi), or the terminal costate ``λ(T)`` (optimal control).
vjp
: Vector-Jacobian product: the kernel that pulls a seed back through one step, from the adjoints of the step's outputs to the adjoints of its inputs. ``hawk.diff.vjp(kernel)`` makes one from a kernel. Each launch node of a graph is reversed by its kernel's vjp.
wrt
: "With respect to": which inputs you want gradients for. ``wrt=None`` (the default) returns them all.
```
