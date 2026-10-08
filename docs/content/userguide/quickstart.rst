Quick start
===========

.. contents:: On this page
   :local:
   :depth: 2

This page walks through a minimal EAGLE program: capture a kernel launch into
a CUDA graph, instantiate a launcher, and replay it.  It mirrors the shipped
``eagle_demo.cu`` smoke test.

.. note::

   New to EAGLE's vocabulary? :doc:`foundations` introduces ``Stream``,
   ``Graph``, ``StreamCapturer``, and ``Launcher`` one at a time before this
   page puts them together. This page's code block is a working confirmation
   of that walkthrough, not a first introduction.

A minimal capture-and-launch (CUDA)
-----------------------------------

The four objects below are introduced in full in :doc:`foundations`; in
short, here's what each does in the program that follows:

- ``Stream`` — an RAII wrapper around a ``cudaStream_t``, the ordered queue
  of GPU work a kernel launch goes onto.
- ``StreamCapturer`` — turns launch recording on (``begin()``) and off
  (``end()``), handing back a ``cudaGraph_t``.
- ``Graph`` — collects captured recordings into a runnable dependency graph.
- ``Launcher`` — the instantiated, replayable graph; ``launch()`` submits it
  again, ``synchronize()`` waits for it to finish.

.. code-block:: cpp

   #include <cstdio>
   #include <vector>
   #include <eagle/eagle.h>

   __global__ void addOne(int* buf, int n)
   {
       int tid = threadIdx.x + blockIdx.x * blockDim.x;
       if (tid < n)
           buf[tid] += 1;
   }

   int main()
   {
       constexpr int N = 64;
       int* dBuf = nullptr;
       cudaMalloc(&dBuf, N * sizeof(int));
       cudaMemset(dBuf, 0, N * sizeof(int));

       // RAII stream + graph builder
       eagle::cuda::Stream stream;
       eagle::cuda::Graph  graph;
       graph.stream(stream.cuda());

       // Capture the kernel launch into the graph
       eagle::cuda::StreamCapturer capturer(stream.cuda());
       capturer.begin();
       addOne<<<1, N, 0, stream.cuda()>>>(dBuf, N);
       graph.addNode(capturer.end());

       // Instantiate and replay
       eagle::cuda::Launcher launcher = graph.launcher();
       launcher.launch();
       launcher.synchronize();

       std::vector<int> host(N, -1);
       cudaMemcpy(host.data(), dBuf, N * sizeof(int), cudaMemcpyDeviceToHost);
       cudaFree(dBuf);
       std::printf("eagle_demo: %s\n", host[0] == 1 ? "OK" : "FAIL");
       return 0;
   }

Building the example
--------------------

.. code-block:: cmake

   find_package(eagle CONFIG REQUIRED)
   add_executable(demo demo.cu)
   target_link_libraries(demo PRIVATE eagle::eagle)

The same source compiles unchanged in ``EAGLE_CPP_MODE`` when written against
the ``eagle::cpu::Host`` dispatcher instead of the CUDA graph launchers — see
:doc:`../modules/module_host`.

Next steps
----------

* :doc:`cuda_graphs` — the graph execution model in depth.
* :doc:`../modules/modules` — a tour of every module.
* :doc:`../api/api_reference` — the full C++ API.

.. seealso::

   :doc:`installation` — install eagle and its aether dependency.
