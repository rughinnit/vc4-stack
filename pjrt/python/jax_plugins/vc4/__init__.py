"""JAX plugin registration for the VideoCore IV QPU engine.

`import jax` discovers this module through the ``jax_plugins`` namespace and
calls :func:`initialize`, which registers the PJRT client in
``libpjrt_vc4.so``.  With ``JAX_PLATFORMS=vc4`` jax then binds to it::

    PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4 python -c 'import jax; print(jax.devices())'

The library is looked up in this order: ``$VC4_PJRT_LIBRARY``, next to this
file, then ``build/libpjrt_vc4.so`` in a source checkout.
"""

import os

import jax._src.xla_bridge as xb

__version__ = "0.1.0"


def _library_path() -> str:
    env = os.environ.get("VC4_PJRT_LIBRARY")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "libpjrt_vc4.so"),
        os.path.join(here, "..", "..", "..", "build", "libpjrt_vc4.so"),
    ):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.join(here, "libpjrt_vc4.so")


def initialize() -> None:
    """Register the vc4 PJRT plugin with jax.  Cheap; no GPU access."""
    path = _library_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"libpjrt_vc4.so not found at {path!r}; build it with "
            "`make -C pjrt` or set VC4_PJRT_LIBRARY"
        )
    xb.register_plugin("vc4", library_path=path, priority=400)
