# The backend's build product is the cp313/aarch64 wheel, which belongs in the
# stack's shared dist/ next to the vc4infer wheel.  The recipe is the same
# script a split checkout uses; it resolves ../infer and ../vc4d from setup.py
# and fails with a message naming what is missing (torch, the archives).
#
#   make -C torch
#   PYTHON=~/work/vc4-stack/.venv/bin/python make -C torch

PYTHON ?= python3

.PHONY: wheel clean

wheel:
	PYTHON=$(PYTHON) ../packaging/build_torch_wheel.sh

clean:
	rm -rf build *.egg-info
