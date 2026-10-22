Plugin schema (manifest and sidecar)
====================================

The on-disk contract between a code generator and eagle's loaders — the
``manifest.json`` a deployment unit carries, the per-kernel sidecar, the
``arg_spec`` role vocabulary, the ``neural_block`` descriptor and the schema
versions — is raptor's schema, and it is documented once, in raptor:
`Plugin schema <https://amasat01.github.io/raptor/content/devguide/plugin_schema.html>`_.

eagle reads that schema through :mod:`eagle.roles` and :mod:`eagle.sidecar`
(both re-export raptor's vocabulary) and :func:`eagle.registry.load_manifest`.

The sidecar schema is v3: ``cells`` replaces ``sample_sums``, and a kernel split in stages adds ``stages``. A cell only a later stage reads is marked ``internal`` in ``cells`` and is not handed back. An eagle that predates v3 refuses a v3 artifact.

float32 on the C++ registry
---------------------------

``eagle::cuda::PluginRegistry`` binds ``double`` data only, so ``from_manifest`` refuses a sidecar that declares
``scalar_type: "float32"`` before it loads anything. The float32 twin of the binding surface is the way out; it is
built once a C++ consumer of a float32 artifact exists. Python plans run float32 today.
