# The workload and the ways of running it


A large class of scientific code integrates many independent problems at
once: one trajectory, one particle or one parameter sample per GPU thread,
each with its own state, its own control flow and its own stopping point.
Written for the GPU, this class tends to lose efficiency in four places:

- **launch overhead**, when every step is a separate kernel launch from the
  host;
- **host round trips**, when the host has to read a device value every step
  to decide whether to continue;
- **finished samples**, when samples that have stopped keep occupying the
  batch and keep being computed;
- **idle lanes**, when a launch is too small to fill the device.

The benchmark is built to exercise all four. It integrates a batch of damped
harmonic oscillators, each with its own frequency, damping ratio and initial
state, using a fixed-step classical Runge-Kutta (RK4) scheme. Each sample
stops after its **own** number of steps. In the *spread* configurations those
step counts are drawn log-uniformly over two decades, so most of the batch
finishes early; in the *uniform* configuration every sample runs the full
count, a dense, all-active batch. The batch size N runs from 10³ to 10⁶.

The step is written once, as a hawk kernel, one sample at a time:

```{literalinclude} ../../../benchmarks/perf_card/perf_card.py
:language: python
:start-after: "# >>> code:hawk_step"
:end-before: "# <<< code:hawk_step"
:dedent: 4
```

`terminated: Terminated` is the per-sample stop mask: a finished sample keeps
its state. `x`, `v` and `k` are read as plain variables holding last launch's
value and reassigned to the next one, which is what updates the state in
place, launch after launch.


## The ways of running it

All arms do the same arithmetic in the same precision (float64) on the same
samples, on the same device; the card's arm table lists each one's loop
structure, and the card records every arm's code.

**hawk + eagle.** The kernel is built with {func}`eagle.deploy` and run until every
sample has finished with {func}`eagle.until_done`: the step, which finishes
its own samples, is captured once as the body of {func}`eagle.repeat_while`,
one CUDA graph whose WHILE node repeats it until the finished count reaches
N. One graph launch runs the whole integration; the host does not read
anything until it is done.

```{literalinclude} ../../../benchmarks/perf_card/perf_card.py
:language: python
:start-after: "# >>> code:eagle_graph"
:end-before: "# <<< code:eagle_graph"
:dedent: 8
```

The plain `@hawk.kernel` decorator builds the automatic kernel. By default
{func}`eagle.until_done` auto-routes it past this device-loop/"band" shape
entirely: at or below the device's own latency-regime capacity it takes one
fused launch of the whole run, above it the persistent (work-stealing)
launch -- both no-graph, no-map, one host-device sync total. The band shown
above (each launch running as many steps per sample as the policy picks, 8
to 64, with the state in registers) is still reachable, and is what this
row forces explicitly so its loop-structure description, byte model and
kernel count stay meaningful; it is also what any artifact falls back to
when a reorder is requested, or when the fast entries are unavailable. The
other hawk + eagle arms vary one thing at a time:

- **compaction**: the single step traced under a kind whose guard reads an
  active-set index map (hawk `Guard(active_set=True)`); every 16 steps, when
  a sample finished since the last time, an {class}`eagle.ActiveSet`
  recomputes the map on the device, inside the same graph, so the launch
  covers only the samples still running;
- **compaction + reorder**: the same loop, plus an occasional physical
  reorder of every per-sample plane once the live samples spread thin over
  their warps (`eagle.ActiveSet(mask, reorder=theta)`), restored to sample
  order at the end;
- **K fused steps per launch**: `@hawk.kernel(steps=16)`, with and without
  compaction (every 64 steps);
- **eagle auto (eagle picks the launch mode)**:
  `@hawk.kernel(steps="auto")` on the active-set kind -- this row is the one
  left on `eagle.until_done`'s own default pick, so it takes the fast,
  no-graph launch described above whenever capacity allows and the band
  (compacting after every launch in which a sample finished) only above it;
  the card records which one each run actually took;
- **hawk + eagle persistent**: the plain automatic kernel forced through
  hawk's persist entry -- one launch, no graph, no map, no policy kernel, a
  grid sized from the SM count in which each lane takes the next unfinished
  sample off a counter whenever its own finishes;
- **eagle.simulate**: the same automatic kernel run by {func}`eagle.simulate`
  (its bound form, `eagle.simulation`, so the repetitions reuse one build),
  given its state and parameters by name -- it auto-routes the same way;
- **eager launch loop**: the single step (`.step`, one step per launch)
  launched from a Python loop with a host read of the finished count after
  every step: the most direct loop, every intermediate state one host read
  away.

```{literalinclude} ../../../benchmarks/perf_card/perf_card.py
:language: python
:start-after: "# >>> code:eagle_simulate"
:end-before: "# <<< code:eagle_simulate"
:dedent: 8
```

**CuPy and PyTorch, masked.** The same arithmetic as array expressions over
the whole batch, with `where` keeping finished samples unchanged and a host
check that some sample is still running after every step; PyTorch also runs
with a block of 16 steps captured as a CUDA graph and replayed. This is how
the computation is commonly written in an array framework: no kernel to
write, and elementwise kernels that stream memory efficiently. Finished
samples are still computed and then discarded.

**JAX and Warp.** JAX compiles a per-sample `lax.while_loop`, batched with
`vmap`, into one executable; Warp runs an explicit per-thread kernel, one
thread per sample, each stopping at its own step.

**CPU OpenMP.** The same hawk kernel: its planes on the host select the host
side of the same `eagle.deploy`, run by hawk + eagle's OpenMP host team on at most
8 threads. It needs no device and no transfers, and it is the correctness
reference the other arms are checked against.
