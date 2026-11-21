import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# The engine tree this backend is built against.  In the monorepo the backend
# lives at torch/, next to infer/ and vc4d/, so resolve both from this file:
# a worktree or a renamed checkout then still builds against its own parent
# tree instead of a hard-coded ~/work path.  The environment overrides stay
# for a split checkout, where the backend is not next to the engine.
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(
    os.environ.get("VC4_ENGINE_DIR", os.path.join(HERE, os.pardir, "infer")))
VC4D = os.path.abspath(
    os.environ.get("VC4D_DIR", os.path.join(HERE, os.pardir, "vc4d")))

# Linked *statically* on purpose: the engine's shared object gained an
# SONAME (libvc4infer.so.1) that is not installed next to it, and a dlopen of
# the plain .so then fails.  The static archive + libvc4client.a remove the
# runtime dependency on that tree altogether (the kernel-blob JIT still uses
# its tools/kgen.py, see csrc/torch_vc4.cpp).
extra_objects = [
    os.path.join(ENGINE, "build", "libvc4infer.a"),
    os.path.join(VC4D, "build", "libvc4client.a"),
]

setup(
    name="torch_vc4",
    version="0.3.0",
    description="Subset PyTorch PrivateUse1 backend for VideoCore IV (VC4)",
    packages=["torch_vc4"],
    ext_modules=[
        CppExtension(
            "torch_vc4._C",
            ["csrc/torch_vc4.cpp"],
            include_dirs=[os.path.join(ENGINE, "include"),
                          os.path.join(VC4D, "include")],
            extra_objects=extra_objects,
            libraries=["m"],
            # -O0/-g0: torch's headers are big; on this 1 GB board the
            # compile peaks near 1 GB of address space (measured), so keep
            # codegen cheap.  The hot code is on the QPU, not here.
            extra_compile_args=["-O0", "-g0", "-std=c++20",
                                "-fno-strict-aliasing",
                                '-DVC4_ENGINE_DIR="%s"' % ENGINE],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
