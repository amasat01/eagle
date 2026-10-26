Seeing a graph from C++: ``eagle/graph``
========================================

.. contents:: On this page
   :local:
   :depth: 2

A graph you cannot see is a graph you cannot trust. The headers under
``eagle/graph/`` give any eagle graph a picture: they turn it into a plain
description (nodes, edges, frames), lay it out in stages, and print it as an
ASCII block diagram, Mermaid or Graphviz DOT. They are header-only, CUDA-free
and cost nothing unless you include them. The Python
:doc:`eagle.Graph <graph_model>` draws itself with exactly this code, so a
C++ graph and a Python graph look the same.

.. code-block:: cpp

   #include <eagle/eagle.h>          // or just <eagle/graph.h>

Describe a graph
----------------

``eagle::graph::describe`` reads the dependencies a native graph already
keeps. It works on ``cpu::Graph``, ``cuda::Graph`` (in a CUDA build),
``compose::GraphComposer`` and, around a body you describe yourself,
``conditional::ConditionalGroup``.

.. literalinclude:: ../../examples/04_graph_view.cpp
   :language: cpp
   :start-after: // [cell:build]
   :end-before: // [cell:build:end]

.. literalinclude:: ../../examples/04_graph_view.cpp
   :language: cpp
   :start-after: // [cell:describe]
   :end-before: // [cell:describe:end]

What it printed
^^^^^^^^^^^^^^^

.. literalinclude:: ../../examples/04_graph_view.cpp.expected.txt
   :language: text

Native nodes carry no plane names, so they are labelled ``native<i>`` and
each edge is a completion edge (``RAW``, no plane). A composer is described
one node per member (its name, or ``member<i>`` when unnamed), chained in
registration order under ``mode="sequenced"``.

The description
---------------

``eagle::graph::Description`` is plain data, so you can also build one by hand
for anything that is not a native graph:

``Node``
  ``id``, ``label``, ``kind``, ``stage`` (the schedule it runs under, or
  empty), ``frame`` (``-1`` for none) and ``opaque``.

``Edge``
  ``src``, ``dst``, ``plane`` and ``kind``: ``RAW``, ``WAR`` or ``WAW``.

``Frame``
  A region around nodes: ``kind`` (``"if"``, ``"while"``, ``"stage"``),
  ``label`` (the text drawn on it) and ``parent`` (``-1`` at top level).

Ids equal positions. ``Description::validate()`` throws
``std::invalid_argument`` naming the first defect: a dangling or self edge,
an id that is not its position, a frame cycle, or a cycle among the edges.

Lay it out
----------

``eagle::graph::layout(description, sweeps = 4)`` puts every node on a stage
(its longest path from a source, so siblings share one) and orders the nodes
inside each stage with barycenter sweeps, keeping the ordering with the fewest
crossings. It is deterministic: the same description always gives the same
``Layout``, whatever order the edges were listed in. ``Layout::xy`` holds unit
grid coordinates (x centred on each stage, y the stage number, growing
downwards) for any renderer you write yourself.

Print it
--------

.. code-block:: cpp

   std::cout << eagle::graph::ascii(d, lay);        // the box diagram
   std::cout << eagle::graph::ascii(d, lay, false); // + - | only, no UTF-8
   std::cout << eagle::graph::mermaid(d);           // flowchart TD
   std::cout << eagle::graph::dot(d);               // Graphviz

The box diagram draws one box per node on its stage, routes the edges with the
plane names under the source, carries an edge that skips a stage as a vertical
line through the stages in between, draws a frame around the nodes of a
conditional with its label, and wraps an opaque node's name in ``<>``. Only ASCII and the light
box-drawing glyphs are drawn, so it lines up in a browser or a log as in a terminal. A
label longer than ``maxLabel`` (24 by default) is cut with an ellipsis.
Mermaid draws frames as subgraphs and DOT as clusters.

Find the edges yourself
-----------------------

If you can tell eagle which arrays each node touches, ``eagle::graph::deriveEdges``
works the dependencies out the way ``eagle.Graph`` does: from the bytes.

.. code-block:: cpp

   std::vector<eagle::graph::PlaneRef> refs = {
       {"x",   0, base_x,   {n}, {8}, 8, /*write*/ 1},
       {"x",   1, base_x,   {n}, {8}, 8, /*read*/  0},   // node 1 reads what node 0 wrote
   };
   auto edges = eagle::graph::deriveEdges(refs);          // one RAW edge, 0 -> 1

A ``PlaneRef`` is the plane's name, the node that touches it, its base address,
shape, byte strides (they may be negative), item size, and how it is touched
(0 read, 1 write, 2 in place). Two planes overlap when they share a byte; the
test is exact for strided views up to ``kOverlapMaxWork`` search steps (so
column views of one matrix do not overlap) and reports an overlap past it.
``siblingConflicts`` takes the same references plus the branches of each
fork/join group and returns every overlap between siblings that has a writer:
the races a graph must refuse.

The busy guard
--------------

``eagle/util/Busy.h`` is the small guard behind Python's :class:`eagle.Busy`,
and it is yours to use for any object that must not be entered twice at once:

.. code-block:: cpp

   eagle::util::BusyGuard guard("MyEngine");
   {
       auto held = guard.scope("launch");   // throws eagle::util::Busy if taken
       // ... a second thread calling guard.scope("build") now gets
       //     "MyEngine.build: another call is still running on this MyEngine;
       //      give each thread its own MyEngine, or wait for it"
   }

It never blocks and never queues: the second caller is told, at once, which
method holds the object.

See also :doc:`../api/api_graph_view` for the reference, and
:doc:`graph_model` for the Python side.
