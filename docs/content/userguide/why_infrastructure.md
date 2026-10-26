# Why scheduling is infrastructure

Graph capture, scheduling, divergent execution and the mapping to hardware belong to eagle, not to the application.

An application states one step of one sample and what each argument is. eagle works out the rest: the graph of nodes and
what depends on what ([Graph model](graph_model)), the recorded replay ([CUDA graphs](cuda_graphs)), what to do when
samples finish at different times ([compaction](../modules/filtering/scan_and_compaction)), and which backend runs it.
The backend is reached only through a versioned C seam (`eagle-backend/1`, see
[Add an execution backend](../devguide/add_an_execution_backend)).

Two things follow. A new domain does not bring its own launch code, so eagle does not grow with the number of
applications. And a scientist does not tune thread layouts to get a divergent ensemble to stay busy.

Code that reimplements any of this inside an example or an application is a sign the infrastructure is missing something.
Report the gap; do not work around it. The review rule is in [How we build](https://amasat01.github.io/raptor/content/principles.html).
