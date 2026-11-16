# SPDX-License-Identifier: Apache-2.0
#
# Builds libeagle_door.so (eagle's C door, host group) and its C test.
#
#   make AETHER_INCLUDE=<aether include dir> WING_FACTS=<raptor wing raptor_facts.h> test
#
# AETHER_INCLUDE: the include directory of an installed aether.
# WING_FACTS:     the cpu wing's generated raptor_facts.h (aether's linalg needs it first).
# OUT:            the build directory (default: build/ here, git-ignored).
#
# Targets: lib (the library), symbols (it exports eg_* names only), test (build
# and run the C test), all (= test), pair-<unit> (see below), pairs (all of them).
#
# pair-<unit>: one generated unit (its kernel text and its generated meta source) built into one unit library and
# run through the door (test/pair_test.c, one row per unit) against the outputs the old path wrote for the same
# case. The units: ss_bcast mat_apply _wide_math _flat rotate mv ksum_c fixed_count. Per unit give
#
#   KERNEL_<unit>=<kernel .cpp>  META_<unit>=<meta .c>  ORACLE_<unit>=<dir with the old path's inputs and outputs>
#
# plus PAIR_WING_DIR=<raptor wing dir>, e.g.
#
#   make ... PAIR_WING_DIR=<dir> KERNEL_mv=<.cpp> META_mv=<.c> ORACLE_mv=<dir> pair-mv
#
# The unit's digest is DIGEST_<unit> (defaults below); its host entry is <unit>_host.
#
# CUDA=<toolkit root> builds the device group into the library (device/Device.cpp without EAGLE_CPU_ONLY,
# linked to the CUDA runtime, which finds the driver at run time); empty: host only. Give the device build its
# own OUT. Device targets build and never run (run their binaries through the GPU pool):
#
#   device              the library, the device fixture unit and door_device_test; run
#                       $(OUT)/door_device_test $(OUT)/libfixture_device.so $(OUT)/libfixture_unit.so
#   pair-device-<unit>  the unit's cuda kernel text (KERNEL_CUDA_<unit>, compiled by the old path's
#                       device recipe, PAIR_DEVICE_FLAGS) and its meta (META_CUDA_<unit>) in one library, plus
#                       pair_device_test; run $(OUT)/pair_device_test $(OUT)/libpair_dev_<unit>.so <oracle dir>
#                       <unit> <digest>. Needs PAIR_CUDA_WING_DIR=<raptor cuda wing dir>. Units: mv ksum_c fixed_count (osc, osc_x4: the
#                       runner pair; _osc, _osc_x4: the active-set runner pair).

CXX ?= g++
CC ?= gcc
OUT ?= $(CURDIR)/build
AETHER_INCLUDE ?=
WING_FACTS ?=

PAIR_WING_DIR ?=
PAIR_UNITS := ss_bcast mat_apply _wide_math _flat rotate mv ksum_c fixed_count
DIGEST_ss_bcast ?= 49a12f72dfd7520aa76ed07e9c4765ed5f4c5a6d4d2b29b5ad19eb91767befec
DIGEST_mat_apply ?= ef89b7e1ef6118d2fe685199b8ae1a656b6aa98dd27b059dec549aea7feea46e
DIGEST__wide_math ?= 5f25d984f02b738ba5b93ec9827d9996f7c6680101014b0db766c4dc9a1d6f09
DIGEST__flat ?= 550e1eff1a17398db9bb909191c0824ef73b562e7d0e28ff773b380765ceb836
DIGEST_rotate ?= be784a85b4cc7f27ccba6cb81a074fb0c7f97779afd26b4798ac71ff8a4ba1d1
DIGEST_mv ?= 834d4da09683429d73c4bb5c92432945c892a1e7c3cd9dbe268d06acb5970cf5
DIGEST_ksum_c ?= 710ca64f2483b5cc197150874c03265a8c10a06011b1b22028cb3e10035db33c
DIGEST_fixed_count ?= 05d3a0e97314e339dbfd5954d37989dbae00c41573028aa809901dc65b49c686
# The installed host unit recipe (the host compiler flags the old path builds a unit with).
PAIR_HOST_FLAGS ?= -O3 -ffp-contract=fast -fno-math-errno -march=native -mprefer-vector-width=512 \
  -fno-trapping-math -DAETHER_HOST_VECTOR_MATH -DAETHER_HVM_MASKED=1 -std=c++23 -fPIC -DAETHER_CPP_MODE

CUDA ?=
NVCC ?= $(CUDA)/bin/nvcc
PAIR_CUDA_WING_DIR ?=
CUDA_WING_FACTS ?= $(PAIR_CUDA_WING_DIR)/raptor_facts.h
PAIR_DEVICE_UNITS := mv osc osc_x4 ksum_c fixed_count _osc _osc_x4 oscillator
# The old path's device recipe for a cuda unit (hawk's nvcc build) in the image format its meta records
# (EG_IMAGE_PTX: -ptx, else -cubin). PAIR_NVCC/PAIR_CCBIN: the compiler the old run used (its cache's
# validity record); PTX of a toolkit newer than the driver does not load.
PAIR_DEVICE_FLAGS ?= -std=c++20 -arch=sm_61 -O3 -Xptxas -O3 -Wno-deprecated-gpu-targets
PAIR_NVCC ?= $(NVCC)
PAIR_CCBIN ?= $(CXX)

ROOT := $(abspath $(CURDIR)/../..)
WARN := -Wall -Wextra -pedantic -Werror
CXXFLAGS_DOOR := -std=c++23 -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
  -ffp-contract=off $(WARN) -fopenmp -DAETHER_CPP_MODE=1 -DEAGLE_CPU_ONLY=1 \
  -include $(WING_FACTS) -I$(ROOT) -I$(AETHER_INCLUDE)
CFLAGS_C := -std=c11 -O2 -ffp-contract=off $(WARN) -I$(ROOT)

comma := ,
CXXFLAGS_DEVICE := $(filter-out -DEAGLE_CPU_ONLY=1,$(CXXFLAGS_DOOR)) -isystem $(CUDA)/include
CUDART := $(if $(CUDA),-L$(CUDA)/lib64 -lcudart -Wl$(comma)-rpath$(comma)$(CUDA)/lib64)

SRC := Context.cpp Unit.cpp Plane.cpp Plan.cpp RunCheck.cpp abi.cpp host/Runner.cpp host/Ops.cpp host/Record.cpp device/Device.cpp
OBJ := $(SRC:%.cpp=$(OUT)/%.o) $(if $(CUDA),$(OUT)/device/Loop.o)
LIB := $(OUT)/libeagle_door.so
FIXTURE := $(OUT)/libfixture_unit.so
TEST := $(OUT)/door_test

.PHONY: all lib symbols test check-inputs check-cuda pairs device $(PAIR_UNITS:%=pair-%) $(PAIR_DEVICE_UNITS:%=pair-device-%)
all: test

check-inputs:
	@test -n "$(AETHER_INCLUDE)" || { echo "set AETHER_INCLUDE"; exit 1; }
	@test -f "$(WING_FACTS)" || { echo "set WING_FACTS to the wing's raptor_facts.h"; exit 1; }

$(OUT)/%.o: %.cpp Context.h eg_door.h eg_unit_meta.h host/Team.h host/Ops.h | check-inputs
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_DOOR) -c $< -o $@

# The device group: built with the CUDA headers when CUDA is set, else its host-only refusals.
$(OUT)/device/Device.o: device/Device.cpp device/Loop.h Context.h eg_door.h eg_unit_meta.h | check-inputs
	@mkdir -p $(dir $@)
	$(CXX) $(if $(CUDA),$(CXXFLAGS_DEVICE),$(CXXFLAGS_DOOR)) -c $< -o $@

# A device runner's captured loop over eagle's capture seam (CUDA builds only).
$(OUT)/device/Loop.o: device/Loop.cu device/Loop.h | check-inputs check-cuda
	@mkdir -p $(dir $@)
	$(NVCC) -std=c++20 -O2 -arch=sm_61 -Wno-deprecated-gpu-targets -Werror all-warnings -ccbin $(CXX) \
	  -Xcompiler -fPIC,-fvisibility=hidden,-Wall,-Wextra,-fopenmp -DAETHER_HAS_CUDA -include $(CUDA_WING_FACTS) -I$(ROOT) -I$(AETHER_INCLUDE) -c $< -o $@

$(LIB): $(OBJ) door.map
	$(CXX) -shared -o $@ $(OBJ) -Wl,--version-script=$(CURDIR)/door.map -Wl,--no-undefined -fopenmp -pthread -ldl $(CUDART)

lib: $(LIB)

# The library exports eg_* names only.
symbols: $(LIB)
	@extra=$$(nm -D --defined-only $(LIB) | awk '{print $$3}' | grep -v '^eg_' || true); \
	if [ -n "$$extra" ]; then echo "FAIL  exports beyond eg_*:"; echo "$$extra"; exit 1; fi; \
	echo "PASS  $(LIB) exports $$(nm -D --defined-only $(LIB) | grep -c ' eg_') eg_* names and nothing else"

# The fixture unit library and three variants of its size block (FIXTURE_BLOCK
# in test/fixture_unit.c): a wrong sizeof(ScalarHandle), none exported, the v1 tag.
FIXTURE_BAD := $(OUT)/libfixture_wrong_size.so $(OUT)/libfixture_no_block.so $(OUT)/libfixture_v1_tag.so
block_of = $(if $(findstring wrong_size,$1),1,$(if $(findstring no_block,$1),2,$(if $(findstring v1_tag,$1),3,0)))

$(FIXTURE) $(FIXTURE_BAD): test/fixture_unit.c eg_door.h eg_unit_meta.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS_C) -DFIXTURE_BLOCK=$(call block_of,$@) -fPIC -shared -fvisibility=hidden -o $@ $<

$(TEST): test/door_test.c eg_door.h $(LIB)
	$(CC) $(CFLAGS_C) -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) -pthread

test: symbols $(FIXTURE) $(FIXTURE_BAD) $(TEST)
	$(TEST) $(FIXTURE) $(FIXTURE_BAD)

PAIR_TEST := $(OUT)/pair_test

$(PAIR_TEST): test/pair_test.c eg_door.h $(LIB)
	$(CC) $(CFLAGS_C) -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) -pthread

# $(1) = unit. The kernel text as the old path compiles it; the meta as C11, pedantic and warning-free.
define PAIR_RULES
$(OUT)/libpair_$(1).so: $$(KERNEL_$(1)) $$(META_$(1)) eg_door.h eg_unit_meta.h | check-inputs
	@test -f "$$(KERNEL_$(1))" -a -f "$$(META_$(1))" -a -d "$$(PAIR_WING_DIR)" || \
	  { echo "set KERNEL_$(1), META_$(1) and PAIR_WING_DIR"; exit 1; }
	@mkdir -p $(OUT)
	$$(CXX) $$(PAIR_HOST_FLAGS) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) \
	  -c $$(KERNEL_$(1)) -o $(OUT)/pair_$(1)_kernel.o
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) \
	  -c $$(META_$(1)) -o $(OUT)/pair_$(1)_meta.o
	$$(CXX) -shared -o $$@ $(OUT)/pair_$(1)_kernel.o $(OUT)/pair_$(1)_meta.o

pair-$(1): symbols $(OUT)/libpair_$(1).so $$(PAIR_TEST)
	@for sym in eg_unit_library_meta $(1)_host; do \
	  nm -D --defined-only $(OUT)/libpair_$(1).so | awk '{print $$$$3}' | grep -qx "$$$$sym" || \
	    { echo "FAIL  $(OUT)/libpair_$(1).so does not export $$$$sym"; exit 1; }; \
	  echo "PASS  $(OUT)/libpair_$(1).so exports $$$$sym"; done
	$$(PAIR_TEST) $(OUT)/libpair_$(1).so $$(ORACLE_$(1)) $(1) $$(DIGEST_$(1))
endef
$(foreach u,$(PAIR_UNITS),$(eval $(call PAIR_RULES,$(u))))

pairs: $(PAIR_UNITS:%=pair-%)

# ---- device targets (CUDA set) ----

check-cuda:
	@test -n "$(CUDA)" -a -x "$(NVCC)" || { echo "set CUDA to a CUDA toolkit root"; exit 1; }

# $(1) the image file, $(2) the symbol, $(3) the object: it exports the image's bytes as $(2) and their count as
# the uint64_t $(2)_size (the two symbols a meta names in device_image and device_image_size).
embed = printf '.section .rodata\n.balign 16\n.globl $(2)\n.type $(2),@object\n$(2):\n.incbin "%s"\n$(2)_end:\n.balign 8\n.globl $(2)_size\n.type $(2)_size,@object\n$(2)_size:\n.quad $(2)_end - $(2)\n.section .note.GNU-stack,"",@progbits\n' $(1) > $(3:.o=.S) && $(CC) -c $(3:.o=.S) -o $(3)

DEVICE_FIXTURE := $(OUT)/libfixture_device.so
DEVICE_TEST := $(OUT)/door_device_test

$(OUT)/fixture_device.fatbin: test/fixture_device.cu | check-cuda
	@mkdir -p $(OUT)
	$(NVCC) -fatbin -arch=sm_61 -Wno-deprecated-gpu-targets -std=c++20 -ccbin $(CXX) -I$(ROOT) $< -o $@

$(DEVICE_FIXTURE): test/fixture_device.c $(OUT)/fixture_device.fatbin eg_door.h eg_unit_meta.h
	$(call embed,$(OUT)/fixture_device.fatbin,fixture__image,$(OUT)/fixture_device_image.o)
	$(CC) $(CFLAGS_C) -fPIC -shared -fvisibility=hidden -o $@ $< $(OUT)/fixture_device_image.o

$(DEVICE_TEST): test/door_device_test.c eg_door.h $(LIB) | check-cuda
	$(CC) $(CFLAGS_C) -isystem $(CUDA)/include -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) $(CUDART) -lm -pthread

device: check-cuda symbols $(FIXTURE) $(DEVICE_FIXTURE) $(DEVICE_TEST)

PAIR_DEVICE_TEST := $(OUT)/pair_device_test
RUNNER_PAIR_TEST := $(OUT)/runner_pair_test
ACTIVE_PAIR_TEST := $(OUT)/active_pair_test
FAST_PAIR_TEST := $(OUT)/fast_pair_test

$(FAST_PAIR_TEST): test/fast_pair_test.c test/pair_util.h eg_door.h $(LIB) | check-cuda
	$(CC) $(CFLAGS_C) -isystem $(CUDA)/include -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door \
	  -Wl,-rpath,$(OUT) $(CUDART) -lm -pthread

$(RUNNER_PAIR_TEST): test/runner_pair_test.c test/pair_util.h eg_door.h $(LIB) | check-cuda
	$(CC) $(CFLAGS_C) -isystem $(CUDA)/include -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door \
	  -Wl,-rpath,$(OUT) $(CUDART) -lm -pthread

$(ACTIVE_PAIR_TEST): test/active_pair_test.c test/pair_util.h eg_door.h $(LIB) | check-cuda
	$(CC) $(CFLAGS_C) -isystem $(CUDA)/include -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door \
	  -Wl,-rpath,$(OUT) $(CUDART) -lm -pthread

$(PAIR_DEVICE_TEST): test/pair_test.c eg_door.h $(LIB) | check-cuda
	$(CC) $(CFLAGS_C) -DEG_PAIR_DEVICE -isystem $(CUDA)/include -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door \
	  -Wl,-rpath,$(OUT) $(CUDART) -lm -pthread

# $(1) = unit. The kernel text compiled as the old path compiles it, embedded as <unit>__image.
define PAIR_DEVICE_RULES
$(OUT)/libpair_dev_$(1).so: $$(KERNEL_CUDA_$(1)) $$(META_CUDA_$(1)) eg_door.h eg_unit_meta.h | check-inputs check-cuda
	@test -f "$$(KERNEL_CUDA_$(1))" -a -f "$$(META_CUDA_$(1))" -a -d "$$(PAIR_CUDA_WING_DIR)" || \
	  { echo "set KERNEL_CUDA_$(1), META_CUDA_$(1) and PAIR_CUDA_WING_DIR"; exit 1; }
	@mkdir -p $(OUT)
	$$(PAIR_NVCC) $$(if $$(shell grep -l EG_IMAGE_PTX $$(META_CUDA_$(1))),-ptx,-cubin) $$(PAIR_DEVICE_FLAGS) \
	  -ccbin $$(PAIR_CCBIN) -I $$(AETHER_INCLUDE) -I $$(AETHER_INCLUDE)/eagle \
	  -I$$(PAIR_CUDA_WING_DIR) $$(KERNEL_CUDA_$(1)) -o $(OUT)/pair_dev_$(1).image
	$$(call embed,$(OUT)/pair_dev_$(1).image,$(1)__image,$(OUT)/pair_dev_$(1)_image.o)
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) \
	  -c $$(META_CUDA_$(1)) -o $(OUT)/pair_dev_$(1)_meta.o
	$$(CC) -shared -o $$@ $(OUT)/pair_dev_$(1)_meta.o $(OUT)/pair_dev_$(1)_image.o

pair-device-$(1): symbols $(OUT)/libpair_dev_$(1).so $$(PAIR_DEVICE_TEST) $$(RUNNER_PAIR_TEST) $$(ACTIVE_PAIR_TEST) $$(FAST_PAIR_TEST)
endef
$(foreach u,$(PAIR_DEVICE_UNITS),$(eval $(call PAIR_DEVICE_RULES,$(u))))
