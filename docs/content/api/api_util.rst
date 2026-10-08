API reference — util
====================

Global typedefs, logging, error handling, observer-pattern primitives,
array slicing, and OpenMP thread-count management.

Global typedefs
---------------

.. doxygenfile:: typedefs.h
   :project: EAGLE

Namespace ``eagle::util``
-------------------------

Includes the nested ``eagle::util::observable`` (the observer-pattern array)
and ``eagle::util::slice`` (array slicing) namespaces — Breathe's ``:members:``
already descends into them, so they are not repeated as separate directives
below (doing so re-registered the same Doxygen anchor IDs twice and tripped
docutils' "Duplicate explicit target name").

.. doxygennamespace:: eagle::util
   :project: EAGLE
   :members:

Error codes — ``eagle::err``
----------------------------

.. doxygenenum:: eagle::err::EagleErrorTypes
   :project: EAGLE
