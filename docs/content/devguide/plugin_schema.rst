Plugin schema (manifest and sidecar)
====================================

The on-disk contract between a code generator and EAGLE's loaders — the
``manifest.json`` a deployment unit carries, the per-kernel sidecar, the
``arg_spec`` role vocabulary, the ``neural_block`` descriptor and the schema
versions — is raptor's schema, and it is documented once, in raptor:
`Plugin schema <https://amasat01.github.io/raptor/content/devguide/plugin_schema.html>`_.

EAGLE reads that schema through :mod:`eagle.roles` and :mod:`eagle.sidecar`
(both re-export raptor's vocabulary) and :func:`eagle.registry.load_manifest`.
