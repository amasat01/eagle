# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# eagle's gates, runnable from this repository alone. CI jobs call these targets.
#
#   make gate       the CPU legs CI runs: ruff + manifesto, the C++ gate (cpp mode, -Werror), every header on
#                   its own with two compilers, the python suite (-m "not gpu and not interop_matrix"), and the
#                   docs (nbcheck, strict Sphinx, public-name inventory, freshness)
#   make gate-gpu   the legs that need a GPU: the C++ gate (cuda mode, -Werror) and the python suite (-m gpu)
#   make gate-all   both
#
# Stages, for the inner loop and for CI jobs that run one leg: gate-lint gate-names gate-cpp gate-py gate-docs gate-cpp-cuda gate-py-gpu
#
# THE ENVIRONMENT. Each run builds its venv under $(GATE_DIR) (default build/gate, git-ignored), fresh, with
# `pip install -e python[test] -c ci/constraints.txt`. GATE_REUSE=1 keeps it (inner loop only).
# SIBLINGS are explicit inputs: RAPTOR_SRC, AETHER_SRC (the repository root), HAWK_SRC, SITE_SRC (the landing
# site, which holds raptor_docs). Unset means the declared dependency from PyPI; AETHER_SRC has no PyPI form for the
# C++ legs and is required there. Nothing else outside this repository is read.
# TOOLCHAINS come from the environment and the gate fails loudly when one is missing: CXX (g++), CUDACXX (nvcc),
# CLANGXX (the second header compiler), and for the docs doxygen on PATH.
# GPU: GPU_RUN is the prefix the GPU runs are launched with (the GPU is held for the run only, never for a build),
# e.g. GPU_RUN="gpu_run.sh --gpu 1 taskset -c 2-7".
# PYTEST_ARGS replaces the default python/tests path (the probe: GATE_REUSE=1 make gate PYTEST_ARGS="python/tests/test_x.py").
# JOBS: GATE_JOBS (default 2) is passed to every build; PYTEST_JOBS (default GATE_JOBS) is the CPU pytest workers, one test
# file per worker; GPU_PYTEST_JOBS (default 3) the GPU ones, sharing the card.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON      ?= python3
GATE_DIR    ?= build/gate
GATE_JOBS   ?= 2
PYTEST_JOBS ?= $(GATE_JOBS)
# GPU rows share the one card: their time is mostly host work, so a few workers overlap it with the device runs
GPU_PYTEST_JOBS ?= 3
GATE_REUSE  ?=
PYTEST_ARGS ?=
RAPTOR_SRC  ?=
AETHER_SRC  ?=
HAWK_SRC    ?=
SITE_SRC    ?=
GPU_RUN     ?=
# The CPU legs compile device code for one architecture and never execute it (CI: sm_70).
CPU_CUDA_ARCH ?= 70
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

comma := ,
GD := $(abspath $(GATE_DIR))
# the notebooks a gate runs write their animations here, not into docs/_static (a gate never changes tracked files)
export DOCS_ANIM_DIR := $(GD)/anim
WARP_CU12_WHEEL := https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl#sha256=6dda3c163dde4edaa7d46ade9fff3ddb1126ee7b147d9d890cd38cc70dafd656
export CMAKE_BUILD_PARALLEL_LEVEL := $(GATE_JOBS)
export PIP_DISABLE_PIP_VERSION_CHECK := 1
CONSTRAINTS := $(CURDIR)/ci/constraints.txt

# a tool the environment must provide: fail with its name, never guess a path
need = @command -v $(1) >/dev/null 2>&1 || { echo "gate: $(2) is not available ($(1)); set it in the environment" >&2; exit 2; }
need_aether = @[ -n "$(AETHER_SRC)" ] && [ -d "$(AETHER_SRC)/aether" ] || { echo "gate: AETHER_SRC (the aether repository root) is required for the C++ legs" >&2; exit 2; }
need_cxx = @[ "$(origin CXX)" != default ] || { echo "gate: CXX is not set in the environment (a g++ for the C++ legs)" >&2; exit 2; }
NVCC = $(or $(CUDACXX),$(if $(CUDA_PATH),$(CUDA_PATH)/bin/nvcc))
# hawk (which builds the kernels the tests deploy) compiles with the same nvcc, never whichever one is on PATH
export HAWK_NVCC ?= $(NVCC)
need_nvcc = @[ -n "$(NVCC)" ] && [ -x "$(NVCC)" ] || { echo "gate: no nvcc: set CUDACXX (or CUDA_PATH with bin/nvcc)" >&2; exit 2; }

.PHONY: help gate gate-gpu gate-all gate-fresh gate-lint gate-names gate-cpp gate-py gate-docs gate-cpp-cuda gate-py-gpu \
        gate-env-cpu gate-env-gpu gate-clean gate-doxygen sanitize-cpp tsan-ft ft-stress nodriver sdist-install release-gate

help:
	@sed -n '4,24p' Makefile

# -- a fresh start ------------------------------------------------------------------------------------------------
gate-clean:
	@[ -n "$(GATE_DIR)" ] && [ "$(GD)" != "/" ] && [ "$(GD)" != "$(CURDIR)" ] || { echo "gate: refusing to remove '$(GD)'" >&2; exit 2; }
	rm -rf "$(GD)"

gate-fresh:
	@if [ -z "$(GATE_REUSE)" ]; then $(MAKE) --no-print-directory gate-clean; fi
	@mkdir -p "$(GD)"

# -- entry points -------------------------------------------------------------------------------------------------
# PYTEST_ARGS given: the inner-loop probe, the python stage alone on those files (use with GATE_REUSE=1)
gate:
ifneq ($(strip $(PYTEST_ARGS)),)
	$(MAKE) --no-print-directory gate-fresh
	$(MAKE) --no-print-directory gate-py
else
	$(MAKE) --no-print-directory gate-fresh
	$(MAKE) --no-print-directory gate-lint
	$(MAKE) --no-print-directory gate-names
	$(MAKE) --no-print-directory gate-cpp
	$(MAKE) --no-print-directory gate-py
	$(MAKE) --no-print-directory gate-docs
endif
	@echo "gate: PASS"

gate-gpu:
	$(MAKE) --no-print-directory gate-fresh
	$(MAKE) --no-print-directory gate-cpp-cuda
	$(MAKE) --no-print-directory gate-py-gpu
	@echo "gate-gpu: PASS"

gate-all:
	$(MAKE) --no-print-directory gate-gpu
	$(MAKE) --no-print-directory gate GATE_REUSE=1

# -- the python environments (cpu: arch $(CPU_CUDA_ARCH), compile-only; gpu: the device's own architecture) --------
# one venv + one editable install per leg; the build tree and the header prefix live under $(GD) too
define ENV_RECIPE
	$(call need_aether)
	$(call need_nvcc)
	$(call need_cxx)
	mkdir -p "$(GD)"
	if [ ! -x "$(GD)/venv_$(1)/bin/python" ]; then $(PYTHON) -m venv "$(GD)/venv_$(1)"; fi
	PIP="$(GD)/venv_$(1)/bin/python -m pip"; \
	$$PIP install --quiet --upgrade pip; \
	$$PIP install --quiet -c "$(CONSTRAINTS)" "scikit-build-core>=0.10" "setuptools-scm>=8" "nanobind>=3.1,<4" pytest numpy; \
	if [ ! -f "$(GD)/prefix_$(1)/share/eagle/cmake/eagleConfig.cmake" ]; then \
	  cmake -DAETHER_CPP_MODE=OFF -DCMAKE_CUDA_COMPILER="$(NVCC)" -DCMAKE_CUDA_HOST_COMPILER="$(CXX)" $(2) -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_$(1)" \
	        -B "$(GD)/aether_$(1)" -S "$(AETHER_SRC)"; \
	  cmake --install "$(GD)/aether_$(1)"; \
	  cmake -B "$(GD)/eagle_hdr_$(1)" -DCMAKE_CUDA_COMPILER="$(NVCC)" -DCMAKE_CUDA_HOST_COMPILER="$(CXX)" $(2) -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_$(1)" .; \
	  cmake --install "$(GD)/eagle_hdr_$(1)"; \
	fi; \
	$(if $(RAPTOR_SRC),$$PIP install --quiet -c "$(CONSTRAINTS)" "$(RAPTOR_SRC)";,) \
	$(if $(SITE_SRC),$$PIP install --quiet --no-deps -c "$(CONSTRAINTS)" "$(SITE_SRC)";,) \
	SKBUILD_BUILD_DIR="$(GD)/sb_dsc_$(1)" $$PIP install --quiet -c "$(CONSTRAINTS)" "$(AETHER_SRC)/dsc"; \
	$(if $(HAWK_SRC),SKBUILD_BUILD_DIR="$(GD)/sb_hawk_$(1)" $$PIP install --quiet -c "$(CONSTRAINTS)" "$(HAWK_SRC)";,) \
	CMAKE_PREFIX_PATH="$(GD)/prefix_$(1):$${CMAKE_PREFIX_PATH:-}" \
	SKBUILD_BUILD_DIR="$(GD)/sb_eagle_$(1)" \
	SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$(NVCC);-DCMAKE_CUDA_HOST_COMPILER=$(CXX);$(3);-DEAGLE_PYTHON_WARN_AS_ERR=ON" \
	$$PIP install --quiet --no-build-isolation --only-binary=:all: -c "$(CONSTRAINTS)" -e "./python[$(4)]"
endef

gate-env-cpu:
	$(call ENV_RECIPE,cpu,-DCMAKE_CUDA_ARCHITECTURES=$(CPU_CUDA_ARCH),-DCMAKE_CUDA_ARCHITECTURES=$(CPU_CUDA_ARCH),test)

# the device's own architecture: CMake's EAGLE_CUDA_ARCHS=native reads it from the driver at configure time
gate-env-gpu:
	$(call ENV_RECIPE,gpu,,-DEAGLE_CUDA_ARCHS=native,test$(comma)cuda12)
	# the profile kit's preflight and comparison arm run Warp: its GPU rows run here, not only on Kaggle. PyPI's
	# warp-lang is built for CUDA 13 and finds no device under a CUDA 12 driver, so the release's CUDA 12 wheel, by hash
	"$(GD)/venv_gpu/bin/python" -m pip install --quiet --no-deps -c "$(CONSTRAINTS)" "warp-lang @ $(WARP_CU12_WHEEL)"

# -- lint ---------------------------------------------------------------------------------------------------------
gate-lint:
	mkdir -p "$(GD)"
	if [ ! -x "$(GD)/venv_lint/bin/python" ]; then $(PYTHON) -m venv "$(GD)/venv_lint"; fi
	"$(GD)/venv_lint/bin/python" -m pip install --quiet -c "$(CONSTRAINTS)" ruff
	"$(GD)/venv_lint/bin/ruff" check
	bash tools/manifesto_check.sh CONTRIBUTING.md

# every public name is named by a test, or on ci/public_names_untested.txt (which may only shrink): raptor's check
gate-names:
	$(if $(RAPTOR_SRC),PYTHONPATH="$(abspath $(RAPTOR_SRC))") $(PYTHON) -m raptor.conformance.public_names --src python --tests python/tests \
	  --allow ci/public_names_untested.txt --module eagle $(if $(COVERAGE_BASE),--base $(COVERAGE_BASE),)

# -- the C++ gate, host mode: the manifest-backed integrity gate and the header self-containment check -----------
gate-cpp:
	$(call need_aether)
	$(call need_cxx)
	@[ -n "$(CLANGXX)" ] && command -v "$(CLANGXX)" >/dev/null || { echo "gate: CLANGXX (the clang++ the header check runs) is not set in the environment" >&2; exit 2; }
	mkdir -p "$(GD)"
	cmake -DAETHER_CPP_MODE=ON -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_cpp" -B "$(GD)/aether_cpp" -S "$(AETHER_SRC)"
	cmake --install "$(GD)/aether_cpp"
	cmake -B "$(GD)/cpp" -DCMAKE_PREFIX_PATH="$(GD)/prefix_cpp" -DFETCHCONTENT_QUIET=ON -DEAGLE_BUILD_TESTS=ON \
	      -DEAGLE_TEST_REPORT=ON -DEAGLE_DEBUG_MODE=ON -DEAGLE_WARN_AS_ERR=ON -DEAGLE_CPP_MODE=ON .
	cmake --build "$(GD)/cpp" -j$(GATE_JOBS)
	tests/check_gate.sh cpp "$(GD)/cpp/tests/eagle_tests"
	tests/check_headers.sh "$(CXX)" "$(GD)/prefix_cpp/include"
	tests/check_headers.sh "$(CLANGXX)" "$(GD)/prefix_cpp/include"

# -- the python suite on the CPU: the compile-only CUDA toolchain builds the extension, nothing runs on a device ---
gate-py: gate-env-cpu
	EAGLE_PLUGIN_EXPECTED_ARCHS=$(CPU_CUDA_ARCH)/$(CPU_CUDA_ARCH) \
	  $(if $(HAWK_SRC),HAWK_AETHER_INCLUDE="$(AETHER_SRC)" HAWK_EAGLE_INCLUDE="$(CURDIR)",) \
	  env -u PYTHONPATH -u LD_LIBRARY_PATH "$(GD)/venv_cpu/bin/python" -m pytest -q -m "not gpu and not interop_matrix" -n $(PYTEST_JOBS) --dist loadfile $(or $(PYTEST_ARGS),python/tests)
# LD_LIBRARY_PATH is dropped: a conda lib dir on it brings its own OpenMP runtime next to the one aether ships
# interop_matrix rows certify a framework (torch, jax, warp, cupy) and FAIL when it is absent, by design: this is the
# one sanctioned deselect, as in CI. gpu rows run in gate-gpu.

# -- the docs: nbcheck runs every notebook on the CPU, so it needs hawk, torch, jax; strict Sphinx is 0 warnings -------
# Doxygen >= 1.10 parses the C++20 `requires` clauses. It comes from the environment (CI installs the pinned official
# release before this runs); the check is first, so an old doxygen fails in seconds, not after the notebooks.
gate-doxygen:
	@command -v doxygen >/dev/null 2>&1 || { echo "gate: doxygen is not on PATH (>= 1.10 needed)" >&2; exit 2; }
	@v="$$(doxygen --version | cut -d' ' -f1)"; [ "$$(printf '%s\n1.10\n' "$$v" | sort -V | head -1)" = 1.10 ] || \
	  { echo "gate: doxygen $$v is too old to parse the C++20 requires clauses; >= 1.10 is needed" >&2; exit 2; }

# the cpu env with the docs extra: the binding is built by the same recipe (prefix, no isolation), never a second way
gate-docs: gate-doxygen
	@[ -n "$(SITE_SRC)" ] && [ -d "$(SITE_SRC)" ] || { echo "gate: SITE_SRC (the landing-site checkout holding raptor_docs) is required for the docs" >&2; exit 2; }
	$(call ENV_RECIPE,cpu,-DCMAKE_CUDA_ARCHITECTURES=$(CPU_CUDA_ARCH),-DCMAKE_CUDA_ARCHITECTURES=$(CPU_CUDA_ARCH),test$(comma)docs)
	PIP="$(GD)/venv_cpu/bin/python -m pip"; \
	$$PIP install --quiet -c "$(CONSTRAINTS)" "$(SITE_SRC)" "jax[cpu]" $(if $(HAWK_SRC),,raptor-hawk) ; \
	$$PIP install --quiet -c "$(CONSTRAINTS)" torch --index-url "$(TORCH_INDEX)"
	export PATH="$(GD)/venv_cpu/bin:$$PATH" $(if $(HAWK_SRC),HAWK_AETHER_INCLUDE="$(AETHER_SRC)" HAWK_EAGLE_INCLUDE="$(CURDIR)",); \
	$(MAKE) -C docs nbcheck; \
	$(MAKE) -C docs strict; \
	python -m raptor_docs.inventory docs/_build/html/objects.inv eagle; \
	python -m raptor_docs.fresh --check --site "$(SITE_SRC)"

# -- the GPU legs -------------------------------------------------------------------------------------------------
gate-cpp-cuda:
	$(call need_aether)
	$(call need_nvcc)
	$(call need_cxx)
	mkdir -p "$(GD)"
	cmake -DAETHER_CPP_MODE=OFF -DCMAKE_CUDA_COMPILER="$(NVCC)" -DCMAKE_CUDA_HOST_COMPILER="$(CXX)" -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_cuda" -B "$(GD)/aether_cuda" -S "$(AETHER_SRC)"
	cmake --install "$(GD)/aether_cuda"
	cmake -B "$(GD)/cuda" -DCMAKE_PREFIX_PATH="$(GD)/prefix_cuda" -DCMAKE_CUDA_COMPILER="$(NVCC)" -DCMAKE_CUDA_HOST_COMPILER="$(CXX)" -DFETCHCONTENT_QUIET=ON \
	      -DEAGLE_BUILD_TESTS=ON -DEAGLE_TEST_REPORT=ON -DEAGLE_DEBUG_MODE=ON -DEAGLE_WARN_AS_ERR=ON -DEAGLE_CUDA_ARCHS=native .
	cmake --build "$(GD)/cuda" -j$(GATE_JOBS)
	$(GPU_RUN) tests/check_gate.sh cuda "$(GD)/cuda/tests/eagle_tests"

# interop_matrix stays deselected as in gate-py: its rows need the frameworks, installed by the Kaggle leg
gate-py-gpu: gate-env-gpu
	$(if $(GPU_RUN),,@echo "gate: GPU_RUN is empty; the GPU rows run on whatever GPU is visible")
	$(if $(AETHER_SRC),EAGLE_TEST_AETHER_INCLUDE="$(AETHER_SRC)",) \
	  $(GPU_RUN) env -u PYTHONPATH -u LD_LIBRARY_PATH "$(GD)/venv_gpu/bin/python" -m pytest -q -m "gpu and not interop_matrix" -n $(GPU_PYTEST_JOBS) --dist loadfile $(or $(PYTEST_ARGS),python/tests)

# ======================================================================================================================
# The CI-only jobs. CI calls these; they are not part of `gate`. The commands are the ones the workflow carried.
# Inputs: AETHER_SRC (the root), RAPTOR_SRC, TSAN_PY_VERSION, BASE_SHA/HEAD_SHA; scratch lives under $(GD).
# ======================================================================================================================

# sanitize-cpp: the CPP_MODE suite under ASan+UBSan through the manifest gate, with three non-vacuity checks
sanitize-cpp:
	$(call need_aether)
	mkdir -p "$(GD)"
	cmake -DAETHER_CPP_MODE=ON -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_san" -B "$(GD)/aether_san" -S "$(AETHER_SRC)"
	cmake --install "$(GD)/aether_san"
	cmake -B "$(GD)/san" -DCMAKE_PREFIX_PATH="$(GD)/prefix_san" -DFETCHCONTENT_QUIET=ON -DEAGLE_BUILD_TESTS=ON \
	      -DEAGLE_DEBUG_MODE=ON -DEAGLE_CPP_MODE=ON -DEAGLE_SANITIZERS=address,undefined .
	cmake --build "$(GD)/san" -j$(GATE_JOBS)
	{ nm -D "$(GD)/san/tests/eagle_tests"; nm "$(GD)/san/tests/eagle_tests"; } 2>/dev/null | grep -q __asan_init \
	  || { echo "RED: eagle_tests carries no __asan_init: this job is not sanitizing"; exit 1; }
	fail=0; \
	for pair in "heap-oob:AddressSanitizer: heap-buffer-overflow" "leak:LeakSanitizer: detected memory leaks" \
	            "uninit:runtime error: load of value" "shift-ub:runtime error: shift exponent"; do \
	  mode="$${pair%%:*}"; banner="$${pair#*:}"; \
	  set +e; ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	    "$(GD)/san/tests/eagle_canary" "$$mode" > "$(GD)/canary.out" 2>&1; rc=$$?; set -e; \
	  if [ "$$rc" -eq 0 ] || ! grep -q "$$banner" "$(GD)/canary.out"; then \
	    echo "RED: canary '$$mode' rc=$$rc, banner '$$banner' present: $$(grep -c "$$banner" "$(GD)/canary.out")"; cat "$(GD)/canary.out"; fail=1; \
	  else echo "canary '$$mode' tripped (rc=$$rc)"; fi; \
	done; exit $$fail
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	  LSAN_OPTIONS="suppressions=$(CURDIR)/tests/sanitize/lsan.supp" tests/check_gate.sh cpp "$(GD)/san/tests/eagle_tests"

# tsan-ft: eagle._core under ThreadSanitizer on a free-threaded CPython built with it (TSAN_PY_VERSION, e.g. 3.14.8)
tsan-ft:
	$(call need_aether)
	@[ -n "$(TSAN_PY_VERSION)" ] || { echo "gate: TSAN_PY_VERSION is not set (the CPython built with ThreadSanitizer)" >&2; exit 2; }
	@[ -n "$(RAPTOR_SRC)" ] || { echo "gate: RAPTOR_SRC is required" >&2; exit 2; }
	mkdir -p "$(GD)"
	tests/sanitize/build_tsan_python.sh "$(TSAN_PY_VERSION)" "$(GD)/tsan-python"
	cmake -DAETHER_CPP_MODE=ON -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_tsan" -B "$(GD)/aether_tsan" -S "$(AETHER_SRC)"
	cmake --install "$(GD)/aether_tsan"
	cmake -B "$(GD)/eagle_hdr_tsan" -S . -DEAGLE_CPP_MODE=ON -DCMAKE_PREFIX_PATH="$(GD)/prefix_tsan" -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_tsan"
	cmake --install "$(GD)/eagle_hdr_tsan"
	CMAKE_PREFIX_PATH="$(GD)/prefix_tsan" RAPTOR_DIR="$(RAPTOR_SRC)" tests/sanitize/tsan_ft.sh "$(GD)/tsan-python/bin/python$(basename $(TSAN_PY_VERSION))t"

# ft-stress: the ft rows, then the whole suite, on a free-threaded interpreter ($(PYTHON) must be one), with torch, jax, warp
ft-stress: gate-env-cpu
	"$(GD)/venv_cpu/bin/python" -c "import sys; assert sys._is_gil_enabled() is False, 'GIL is enabled'; print(sys.version)"
	[ -z "$${PYTHON_GIL:-}" ] || { echo "PYTHON_GIL is set"; exit 1; }
	PIP="$(GD)/venv_cpu/bin/python -m pip"; \
	$$PIP install --quiet --only-binary=:all: -c "$(CONSTRAINTS)" torch --index-url "$(TORCH_INDEX)" --extra-index-url https://pypi.org/simple || echo "no torch wheel for this Python: its rows skip"; \
	$$PIP install --quiet --only-binary=:all: -c "$(CONSTRAINTS)" "jax[cpu]" || echo "no jax wheel for this Python: its rows skip"; \
	$$PIP install --quiet --only-binary=:all: -c "$(CONSTRAINTS)" warp-lang || echo "no warp wheel for this Python: its rows skip"
	export EAGLE_PLUGIN_EXPECTED_ARCHS=$(CPU_CUDA_ARCH)/$(CPU_CUDA_ARCH); PY="$(GD)/venv_cpu/bin/python"; \
	N="$$($$PY -m pytest python/tests -m ft --collect-only -q -p no:cacheprovider | grep -c '::' || true)"; \
	echo "ft rows collected: $$N"; [ "$$N" -gt 0 ] || { echo "ft selection collected zero tests"; exit 1; }; \
	$$PY -m pytest python/tests -m ft -q -rs -p no:cacheprovider -W error::RuntimeWarning; \
	$$PY -m pytest python/tests -q -p no:cacheprovider -W error::RuntimeWarning -m "not gpu and not interop_matrix"

# nodriver: the core-only wheel (no plugin, no nvcc) in a fresh venv; imports, the typed refusal, then a copy of the suite
nodriver:
	$(call need_aether)
	@[ -n "$(RAPTOR_SRC)" ] || { echo "gate: RAPTOR_SRC is required" >&2; exit 2; }
	if ldconfig -p | grep -q 'libcuda\.so\.1'; then echo "libcuda.so.1 present"; exit 1; fi
	if command -v nvcc; then echo "nvcc present"; exit 1; fi
	mkdir -p "$(GD)"
	cmake -DAETHER_CPP_MODE=ON -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_nd" -B "$(GD)/aether_nd" -S "$(AETHER_SRC)"
	cmake --install "$(GD)/aether_nd"
	cmake -B "$(GD)/eagle_hdr_nd" -S . -DEAGLE_CPP_MODE=ON -DCMAKE_PREFIX_PATH="$(GD)/prefix_nd" -DCMAKE_INSTALL_PREFIX="$(GD)/prefix_nd"
	cmake --install "$(GD)/eagle_hdr_nd"
	$(PYTHON) -m venv "$(GD)/venv_nd"
	PIP="$(GD)/venv_nd/bin/python -m pip"; \
	$$PIP install --quiet --upgrade pip pytest numpy build scikit-build-core setuptools-scm nanobind; \
	$$PIP install "$(RAPTOR_SRC)" "$(AETHER_SRC)/dsc"; \
	CMAKE_PREFIX_PATH="$(GD)/prefix_nd" "$(GD)/venv_nd/bin/python" -m build --wheel --no-isolation --outdir "$(GD)/dist_nd" \
	  -Ccmake.define.EAGLE_PYTHON_CUDA_PLUGIN=OFF -Ccmake.define.EAGLE_PYTHON_MPI=OFF python; \
	if unzip -l "$(GD)"/dist_nd/*.whl | grep -q libeagle_cuda; then echo "plugin in a core-only wheel"; exit 1; fi; \
	$$PIP install "$(GD)"/dist_nd/*.whl
	cd "$(GD)" && "$(GD)/venv_nd/bin/python" "$(CURDIR)/tools/ci_import_check.py" nodriver
	mkdir -p "$(GD)/suite"
	cp -rL python/tests python/conftest.py "$(GD)/suite/"
	cd "$(GD)/suite" && RAPTOR_ROOT="$(RAPTOR_SRC)" "$(GD)/venv_nd/bin/python" -m pytest tests -q -p no:cacheprovider -W error::pytest.PytestUnknownMarkWarning \
	  -m "not gpu and not gpu2 and not torch and not interop_matrix and not repo_local"
	cd "$(GD)/suite" && "$(GD)/venv_nd/bin/python" -c 'import eagle; assert "site-packages" in eagle.__file__, eagle.__file__'

# sdist-install: the sdist installs into a clean venv with warnings as errors, imports and computes
sdist-install:
	@[ -n "$(RAPTOR_SRC)" ] && [ -n "$(AETHER_SRC)" ] || { echo "gate: RAPTOR_SRC and AETHER_SRC are required" >&2; exit 2; }
	mkdir -p "$(GD)"
	$(PYTHON) -m venv "$(GD)/venv_sd"
	"$(GD)/venv_sd/bin/python" -m pip install --quiet build
	cd python && "$(GD)/venv_sd/bin/python" -m build --sdist --outdir "$(GD)/dist_sd"
	PIP="$(GD)/venv_sd/bin/python -m pip"; \
	$$PIP install --quiet --upgrade pip; \
	$$PIP install --quiet "$(RAPTOR_SRC)" "$(AETHER_SRC)/dsc"; \
	$$PIP install -v -Ccmake.define.EAGLE_PYTHON_WARN_AS_ERR=ON "$(GD)"/dist_sd/raptor_eagle-*.tar.gz; \
	$$PIP check
	cd "$(GD)" && "$(GD)/venv_sd/bin/python" "$(CURDIR)/tools/ci_import_check.py" sdist

# release-gate: the PR label, the diff, the body and the tag agree. BASE_SHA and HEAD_SHA name the commits; the label
# and body come in as PR_LABELS and PR_BODY, as the script reads them
release-gate:
	@[ -n "$(BASE_SHA)" ] && [ -n "$(HEAD_SHA)" ] || { echo "gate: BASE_SHA and HEAD_SHA are required" >&2; exit 2; }
	bash tools/release_gate.sh "$(BASE_SHA)" "$(HEAD_SHA)"
