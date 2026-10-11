C5 — Print a graph
==================

    Turn any eagle graph into a picture: the same layout and renderers draw
    a C++ ``cpu::Graph`` and a Python ``eagle.Graph``.

**Time:** ~2 min · **Runs on:** CPU (pure C++, no CUDA) · **You need:** C2

.. literalinclude:: ../../../examples/04_graph_view.cpp
   :language: cpp
   :start-after: // [cell:describe]
   :end-before: // [cell:describe:end]

.. literalinclude:: ../../../examples/04_graph_view.cpp.expected.txt
   :language: text

``describe`` read the four nodes and their dependencies; ``layout`` put the two
siblings on one stage; ``ascii`` drew the boxes. The full story, including
Mermaid, DOT and finding edges from raw memory ranges, is
:doc:`../graph_view`.
