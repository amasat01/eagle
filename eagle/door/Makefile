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
PAIR_DEVICE_UNITS := mv osc osc_x4 ksum_c fixed_count _osc _osc_x4 _gosc _plain kg_fwd kq oscillator bosc
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

SRC := Context.cpp Unit.cpp Plane.cpp Plan.cpp RunCheck.cpp abi.cpp host/Runner.cpp host/Reverse.cpp host/Ops.cpp host/Record.cpp device/Device.cpp
OBJ := $(SRC:%.cpp=$(OUT)/%.o) $(if $(CUDA),$(OUT)/device/Loop.o)
LIB := $(OUT)/libeagle_door.so
FIXTURE := $(OUT)/libfixture_unit.so
TEST := $(OUT)/door_test

.PHONY: all lib symbols test check-inputs check-cuda pairs pair-reverse pair-reverse-active pair-reverse-timed pair-reverse-fused pair-reverse-kstep pair-reverse-kstep3 pair-reverse-kstep7 pair-reverse-kstep8 pair-reverse-reorder pair-reverse-t0 pair-reverse-counter pair-refine pair-refine-reverse pair-refine-pend pair-refine-reverse-pend pair-refine-reverse-grid pair-refine-reverse-grid-refused pair-status-grid pair-status-tangent pair-status-two device pair-device-reverse pair-device-refine pair-device-refine-reverse $(PAIR_UNITS:%=pair-%) $(PAIR_DEVICE_UNITS:%=pair-device-%)
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
	  -Xcompiler -fPIC,-fvisibility=hidden,-Wall,-Wextra,-fopenmp -DAETHER_HAS_CUDA -diag-suppress=20050 -include $(CUDA_WING_FACTS) -I$(ROOT) -I$(AETHER_INCLUDE) -c $< -o $@

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

# eg_reverse: the closed-form fixture library (test/fixture_reverse.c) and its test.
REV_FIXTURE := $(OUT)/libfixture_reverse.so
REV_TEST := $(OUT)/reverse_test

$(REV_FIXTURE): test/fixture_reverse.c eg_door.h eg_unit_meta.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS_C) -fPIC -shared -fvisibility=hidden -o $@ $<

$(REV_TEST): test/reverse_test.c test/pair_util.h eg_door.h $(LIB)
	$(CC) $(CFLAGS_C) -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) -lm -pthread

test: symbols $(FIXTURE) $(FIXTURE_BAD) $(TEST) $(REV_FIXTURE) $(REV_TEST)
	$(TEST) $(FIXTURE) $(FIXTURE_BAD)
	$(REV_TEST) $(REV_FIXTURE)

# pair-reverse: rr_pend, its one-step taping twin and its step vjp (the old path's kernel texts, KERNEL_rr_pend,
# KERNEL_rr_pend_tape, KERNEL_rr_pend_step_vjp) in one library with META_reverse (their metas and the derived records),
# run by reverse_test against ORACLE_reverse (the old path's states and gradients). The twin's and the vjp's size
# blocks are renamed: the primal's is the library's.
DIGEST_rr_pend ?= 3bc7bb7564e5b3bf766bbdca405dfce7a30b422d2439bb135f171868307c905a
DIGEST_rr_pend_tape ?= a52f08cd753ec50120a28a2aae5a0ed87515d21f573a52a31d8ea34633895e78
DIGEST_rr_pend_step_vjp ?= 59410b1d2c471e7f9f34e5a7e4e2d8bc2ade1a32faa942246555dd144193f0fa
REV_RENAME = -Deagle_abi_tag=$(1)_abi_tag -Deagle_layout_sizes=$(1)_layout_sizes

# $(1) = the meta (META_reverse / META_act), $(2) = the library name, $(3) = the primal's kernel (rr_pend / pend_body).
# The derived records the door loaded, stated by hand until the library emits them.
META_reverse ?= test/derived/rr_pend_meta.c
META_act ?= test/derived/pend_body_meta.c
META_act_t0 ?= test/derived/pend_body_time0_meta.c
META_act_counter ?= test/derived/pend_body_counter_meta.c
META_timed ?= test/derived/timed_meta.c
META_auto ?= test/derived/rr_pend_auto_meta.c
define REV_RULES
$(OUT)/libpair_$(2).so: $$(KERNEL_$(3)) $$(KERNEL_$(3)_tape) $$(KERNEL_$(3)_step_vjp) $$(META_$(1)) eg_door.h eg_unit_meta.h | check-inputs
	@test -f "$$(KERNEL_$(3))" -a -f "$$(KERNEL_$(3)_tape)" -a -f "$$(KERNEL_$(3)_step_vjp)" -a -f "$$(META_$(1))" -a \
	  -d "$$(PAIR_WING_DIR)" || { echo "set KERNEL_$(3), KERNEL_$(3)_tape, KERNEL_$(3)_step_vjp, META_$(1) and PAIR_WING_DIR"; exit 1; }
	@mkdir -p $$(OUT)
	$$(CXX) $$(PAIR_HOST_FLAGS) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_$(3)) -o $$(OUT)/$(2)_primal.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,tape) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) \
	  -c $$(KERNEL_$(3)_tape) -o $$(OUT)/$(2)_tape.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,vjp) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) \
	  -c $$(KERNEL_$(3)_step_vjp) -o $$(OUT)/$(2)_vjp.o
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) -c $$(META_$(1)) -o $$(OUT)/$(2)_meta.o
	$$(CXX) -shared -o $$@ $$(OUT)/$(2)_primal.o $$(OUT)/$(2)_tape.o $$(OUT)/$(2)_vjp.o $$(OUT)/$(2)_meta.o
endef
$(eval $(call REV_RULES,reverse,reverse,rr_pend))
$(eval $(call REV_RULES,act,reverse_active,pend_body))
$(eval $(call REV_RULES,timed,reverse_timed,timed))
$(eval $(call REV_RULES,auto,reverse_auto,rr_pend_auto))
$(eval $(call REV_RULES,act_t0,reverse_t0,pend_body))
$(eval $(call REV_RULES,act_counter,reverse_counter,pend_body))

pair-reverse: symbols $(OUT)/libpair_reverse.so $(REV_TEST)
	$(REV_TEST) $(REV_FIXTURE) $(OUT)/libpair_reverse.so $(ORACLE_reverse) $(DIGEST_rr_pend) $(DIGEST_rr_pend_tape) \
	  $(DIGEST_rr_pend_step_vjp)

# pair-reverse-active: _pend_body (the old path's rr_pend_map, an active-set step) compacting every 4 and every 16 steps
# while recording; its states and gradients equal the old path's bit for bit (ORACLE_act, ORACLE_act16).
DIGEST_pend_body ?= a1929a54693dcbd5c69c818d24a4a3240cb635d10ca63bbaaa2216ca3d86914e
DIGEST_pend_body_tape ?= 884913ccc546f5b55fb48a9854818c62f07596a9b79e9ca1c4d2820041b26cc2
DIGEST_pend_body_step_vjp ?= 59410b1d2c471e7f9f34e5a7e4e2d8bc2ade1a32faa942246555dd144193f0fa
# pair-reverse-timed: _timed (the old path's rr_timed: the time plane in index form, a start copy t_0, dt read as a plane eagle
# fills, the step index as a plane eagle fills), window 37; states and gradients equal the old path's bit for bit (ORACLE_timed).
DIGEST_timed ?= c65e6c2482a9fa9db1d3e63db3422cebd995cefda7ebde5b29924f4a8ff1ddc5
DIGEST_timed_tape ?= a7f940e9d26d43728ae07ff682ddda4fa1a5c77df1ca49c795e65a67e611087e
DIGEST_timed_step_vjp ?= 7e96064e7bac22b669c0eb9fbcc145e7b9b9b55d45a4279688d213d12ef1e752
pair-reverse-timed: symbols $(OUT)/libpair_reverse_timed.so $(REV_TEST)
	$(REV_TEST) $(REV_FIXTURE) $(OUT)/libpair_reverse_timed.so $(ORACLE_timed) $(DIGEST_timed) $(DIGEST_timed_tape) \
	  $(DIGEST_timed_step_vjp) 0 _timed

# pair-reverse-fused: rr_pend_auto (the old path's automatic kernel: a K-step taping twin, EG_DERIVED_TAPE_FUSED, replayed by the same
# fused entry, at most 64 steps per launch and never across a window) over the one-step step vjp; window 400 (built in); states and
# gradients equal the old path's bit for bit (ORACLE_auto).
DIGEST_rr_pend_auto ?= 3bc7bb7564e5b3bf766bbdca405dfce7a30b422d2439bb135f171868307c905a
DIGEST_rr_pend_auto_tape ?= 9a51d0e7bc48d0c09cb3d6b62cd520d58d64b221d707d6ea431584627f1cd8ef
DIGEST_rr_pend_auto_step_vjp ?= 59410b1d2c471e7f9f34e5a7e4e2d8bc2ade1a32faa942246555dd144193f0fa
pair-reverse-fused: symbols $(OUT)/libpair_reverse_auto.so $(REV_TEST)
	$(REV_TEST) $(REV_FIXTURE) $(OUT)/libpair_reverse_auto.so $(ORACLE_auto) $(DIGEST_rr_pend_auto) $(DIGEST_rr_pend_auto_tape) \
	  $(DIGEST_rr_pend_auto_step_vjp) 0 rr_pend_auto

# pair-reverse-kstep<K>: steps_reverse FUSED: hawk.steps(rr_pend, K) (the K-step primal rr_pend_x<K>, window K), its K-step taping twin,
# step vjp (short windows) and K-step vjp (every full window one launch that recomputes its steps); states, gradients, steps and
# launches equal the old path's fused run bit for bit (ORACLE_kstep<K>). K = 4 is "kstep", K = 3, 7 and 8 "kstep3", "kstep7", "kstep8".
DIGEST_kstep ?= 5e1b2ec104ef5491afe94742b85cf99ddb210ec91beb36b1a3101aaf6bbeead3
DIGEST_kstep_tape ?= 6dbf7f16cb41315bf79470f5180a0c4e478da08885360dc87f07c4e6025e8c49
DIGEST_kstep_fused_vjp ?= 569995a5df16dd112d0c91fd9ff75ea85f48be532ba88cbe32b4568b3e8273f9
META_kstep ?= test/derived/rr_pend_fused_meta.c
DIGEST_kstep3 ?= f5ee5386c498b3659c5a145b2bb9d07250c00917ef5a89aaf2817d2bf34727ec
DIGEST_kstep3_tape ?= 3d4973d066cf7f2fe1e868b46a408e2f0d2a66507b0ef623eac5f387e3f7d3e1
DIGEST_kstep3_fused_vjp ?= 486e06a4ac6b16252c8db348692ac5ddc1ac596d4833884981a19923fb756197
META_kstep3 ?= test/derived/rr_pend_fused3_meta.c
DIGEST_kstep7 ?= 12393cfab3ed5feb30843f52f2acf75e4064d61db5ee3d8e1d1d73122f926ee8
DIGEST_kstep7_tape ?= 47605e0703515043c53991d8fd4cc94cb9cd81b27cda6ae952d51e1a925948f2
DIGEST_kstep7_fused_vjp ?= f93873fc2595ffe731dfa2513f032611803e0aec02357a752e7301194b18d1f6
META_kstep7 ?= test/derived/rr_pend_fused7_meta.c
DIGEST_kstep8 ?= a56d3b44588a7389ba9f49da7ba55bd9d62af80a2e6550ce478dbace8faded3f
DIGEST_kstep8_tape ?= 2d20ffd3c860e37349593dba91964bece85bf2bd2b20a2bf73a9a79fb17a2c62
DIGEST_kstep8_fused_vjp ?= 811557ca36771b0d1f3f4727f91fe0c5c49cffc8620afd24cb44c626406e5568
META_kstep8 ?= test/derived/rr_pend_fused8_meta.c
define KSTEP_RULES
$(OUT)/libpair_$(1).so: $$(KERNEL_$(1)) $$(KERNEL_$(1)_tape) $$(KERNEL_$(1)_step_vjp) $$(KERNEL_$(1)_fused_vjp) $$(META_$(1)) eg_door.h eg_unit_meta.h | check-inputs
	@test -f "$$(KERNEL_$(1))" -a -f "$$(KERNEL_$(1)_tape)" -a -f "$$(KERNEL_$(1)_step_vjp)" -a -f "$$(KERNEL_$(1)_fused_vjp)" -a -f "$$(META_$(1))" -a \
	  -d "$$(PAIR_WING_DIR)" || { echo "set KERNEL_$(1), _tape, _step_vjp, _fused_vjp, META_$(1) and PAIR_WING_DIR"; exit 1; }
	@mkdir -p $(OUT)
	$$(CXX) $$(PAIR_HOST_FLAGS) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_$(1)) -o $(OUT)/$(1)_primal.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,tape) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) \
	  -c $$(KERNEL_$(1)_tape) -o $(OUT)/$(1)_tape.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,vjp) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) \
	  -c $$(KERNEL_$(1)_step_vjp) -o $(OUT)/$(1)_vjp.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,fvjp) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) \
	  -c $$(KERNEL_$(1)_fused_vjp) -o $(OUT)/$(1)_fvjp.o
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) -c $$(META_$(1)) -o $(OUT)/$(1)_meta.o
	$$(CXX) -shared -o $$@ $(OUT)/$(1)_primal.o $(OUT)/$(1)_tape.o $(OUT)/$(1)_vjp.o $(OUT)/$(1)_fvjp.o $(OUT)/$(1)_meta.o

pair-reverse-$(1): symbols $(OUT)/libpair_$(1).so $$(REV_TEST)
	$$(REV_TEST) $$(REV_FIXTURE) $(OUT)/libpair_$(1).so $$(ORACLE_$(1)) $$(DIGEST_$(1)) $$(DIGEST_$(1)_tape) \
	  $$(DIGEST_rr_pend_step_vjp) 0 $(2) 64 fused $$(DIGEST_$(1)_fused_vjp)
endef
$(eval $(call KSTEP_RULES,kstep,rr_pend_x4))
$(eval $(call KSTEP_RULES,kstep3,rr_pend_x3))
$(eval $(call KSTEP_RULES,kstep7,rr_pend_x7))
$(eval $(call KSTEP_RULES,kstep8,rr_pend_x8))

# pair-reverse-reorder: the same unit over 4096 samples with EG_RUN_REORDER (the old path's reordering run, every 4, theta 0.5).
pair-reverse-reorder: symbols $(OUT)/libpair_reverse_active.so $(REV_TEST)
	$(REV_TEST) $(REV_FIXTURE) $(OUT)/libpair_reverse_active.so $(ORACLE_reorder) $(DIGEST_pend_body) $(DIGEST_pend_body_tape) \
	  $(DIGEST_pend_body_step_vjp) 4 _pend_body 4096

# pair-reverse-t0: the reordering run of a twin that holds a start copy is refused by name (no old-path oracle).
pair-reverse-t0: symbols $(OUT)/libpair_reverse_t0.so $(REV_TEST)
	$(REV_TEST) $(REV_FIXTURE) $(OUT)/libpair_reverse_t0.so $(ORACLE_reorder) $(DIGEST_pend_body) $(DIGEST_pend_body_tape) \
	  $(DIGEST_pend_body_step_vjp) 4 _pend_body 4096 refuse

# pair-reverse-counter: a link on the twin's finish counter (eagle's own slot) is refused by name.
pair-reverse-counter: symbols $(OUT)/libpair_reverse_counter.so $(REV_TEST)
	$(REV_TEST) $(REV_FIXTURE) $(OUT)/libpair_reverse_counter.so $(ORACLE_reorder) $(DIGEST_pend_body) $(DIGEST_pend_body_tape) \
	  $(DIGEST_pend_body_step_vjp) 4 _pend_body 4096 counter

pair-reverse-active: symbols $(OUT)/libpair_reverse_active.so $(REV_TEST)
	$(REV_TEST) $(REV_FIXTURE) $(OUT)/libpair_reverse_active.so $(ORACLE_act) $(DIGEST_pend_body) $(DIGEST_pend_body_tape) \
	  $(DIGEST_pend_body_step_vjp) 4 _pend_body
	$(REV_TEST) $(REV_FIXTURE) $(OUT)/libpair_reverse_active.so $(ORACLE_act16) $(DIGEST_pend_body) $(DIGEST_pend_body_tape) \
	  $(DIGEST_pend_body_step_vjp) 16 _pend_body

# pair-refine: ev_osc (the old path's oscillator with a stop rule), its one-step taping twin, its Newton and its state guard
# vjp (KERNEL_ev, KERNEL_ev_tape, KERNEL_ev_newton, KERNEL_ev_guard) in one library with META_ev, run by refine_test with
# events REFINE against ORACLE_ev (the old path's end states, stop times and statuses; ORACLE_ev_zero: max_steps 3 with
# on_no_event zero). The twin's, Newton's and guard's size blocks are renamed: the primal's is the library's.
META_ev ?= test/derived/ev_osc_meta.c
DIGEST_ev ?= 9ec2e13d142fdf75dc88fcc3888af20de98d3389f874905220fcf72f524ccc9c
DIGEST_ev_tape ?= d56d34e9962e31adbe26a0b8a27b9a7939355bd52d06e86db0710a5599a07970
DIGEST_ev_newton ?= 414905492f406483a64e202fc6cab04ec2b2dccd4c2e21c6d510d47cb4c780c6
DIGEST_ev_guard ?= 012d7e9df31a4822327f8c88a45d0bfdbe5f8f3dfba0871c0edd02ff073028bc
DIGEST_ev_step_vjp ?= 2209bcec8684c21e2d0d232238b8e31f6b6d7082dc490e4cba1dd2b562099d0d
DIGEST_ev_refine_vjp ?= d463a94a44d802220095aa54ea462f827e19aeb76a9a7bb226a51f98cec0247b
REFINE_TEST := $(OUT)/refine_test

$(REFINE_TEST): test/refine_test.c test/pair_util.h eg_door.h $(LIB)
	$(CC) $(CFLAGS_C) -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) -lm -pthread

$(OUT)/libpair_refine.so: $(KERNEL_ev) $(KERNEL_ev_tape) $(KERNEL_ev_newton) $(KERNEL_ev_guard) $(KERNEL_ev_step_vjp) $(KERNEL_ev_refine_vjp) $(META_ev) eg_door.h eg_unit_meta.h | check-inputs
	@test -f "$(KERNEL_ev)" -a -f "$(KERNEL_ev_tape)" -a -f "$(KERNEL_ev_newton)" -a -f "$(KERNEL_ev_guard)" -a -f "$(KERNEL_ev_step_vjp)" -a -f "$(KERNEL_ev_refine_vjp)" -a -f "$(META_ev)" -a \
	  -d "$(PAIR_WING_DIR)" || { echo "set KERNEL_ev, KERNEL_ev_tape, KERNEL_ev_newton, KERNEL_ev_guard, KERNEL_ev_step_vjp, KERNEL_ev_refine_vjp, META_ev and PAIR_WING_DIR"; exit 1; }
	@mkdir -p $(OUT)
	$(CXX) $(PAIR_HOST_FLAGS) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_ev) -o $(OUT)/refine_primal.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,tape) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) \
	  -c $(KERNEL_ev_tape) -o $(OUT)/refine_tape.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,newton) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) \
	  -c $(KERNEL_ev_newton) -o $(OUT)/refine_newton.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,guard) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) \
	  -c $(KERNEL_ev_guard) -o $(OUT)/refine_guard.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,stepvjp) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) \
	  -c $(KERNEL_ev_step_vjp) -o $(OUT)/refine_step_vjp.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,refvjp) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) \
	  -c $(KERNEL_ev_refine_vjp) -o $(OUT)/refine_refvjp.o
	$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$(ROOT) -c $(META_ev) -o $(OUT)/refine_meta.o
	$(CXX) -shared -o $@ $(OUT)/refine_primal.o $(OUT)/refine_tape.o $(OUT)/refine_newton.o $(OUT)/refine_guard.o $(OUT)/refine_step_vjp.o $(OUT)/refine_refvjp.o $(OUT)/refine_meta.o
	$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$(ROOT) -DEV_OSC_N_EVENTS=2 -c $(META_ev) -o $(OUT)/refine_meta_n2.o
	$(CXX) -shared -o $(OUT)/libpair_refine_n2.so $(OUT)/refine_primal.o $(OUT)/refine_tape.o $(OUT)/refine_newton.o $(OUT)/refine_guard.o $(OUT)/refine_step_vjp.o $(OUT)/refine_refvjp.o $(OUT)/refine_meta_n2.o

pair-refine: symbols $(OUT)/libpair_refine.so $(REFINE_TEST)
	$(REFINE_TEST) $(OUT)/libpair_refine.so $(ORACLE_ev) $(DIGEST_ev) $(DIGEST_ev_tape) $(DIGEST_ev_newton) $(DIGEST_ev_guard) 400 0 1
	$(REFINE_TEST) $(OUT)/libpair_refine.so $(ORACLE_ev_zero) $(DIGEST_ev) $(DIGEST_ev_tape) $(DIGEST_ev_newton) $(DIGEST_ev_guard) 3 1 1
	$(REFINE_TEST) $(OUT)/libpair_refine.so $(ORACLE_ev_zero) $(DIGEST_ev) $(DIGEST_ev_tape) $(DIGEST_ev_newton) $(DIGEST_ev_guard) 3 0 0
	$(REFINE_TEST) $(OUT)/libpair_refine_n2.so $(ORACLE_ev_zero) $(DIGEST_ev) $(DIGEST_ev_tape) $(DIGEST_ev_newton) $(DIGEST_ev_guard) 3 9 0

REFINE_REVERSE_TEST := $(OUT)/refine_reverse_test

$(REFINE_REVERSE_TEST): test/refine_reverse_test.c test/pair_util.h eg_door.h $(LIB)
	$(CC) $(CFLAGS_C) -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) -lm -pthread

# pair-refine-reverse: the recorded refine run and its reverse over the same library against ORACLE_evrr (the old path's
# gradients for seeds t, v, x: test_events_sensitivity_fd's reverse on ev_osc, cpu).
pair-refine-reverse: symbols $(OUT)/libpair_refine.so $(REFINE_REVERSE_TEST)
	$(REFINE_REVERSE_TEST) $(OUT)/libpair_refine.so $(ORACLE_evrr) $(DIGEST_ev) $(DIGEST_ev_tape) $(DIGEST_ev_newton) $(DIGEST_ev_guard) \
	  $(DIGEST_ev_step_vjp) $(DIGEST_ev_refine_vjp)

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

# The same kernel as nvcc 12.9 images (test/fixture_device.cu): a fatbin with sm_61 code and compute_61 PTX, PTX only, and an sm_70 cubin.
$(OUT)/fat129.fatbin: test/fixture_device.cu | check-cuda
	@mkdir -p $(OUT)
	$(NVCC) -fatbin -gencode arch=compute_61,code=[sm_61,compute_61] -Wno-deprecated-gpu-targets -std=c++20 -ccbin $(CXX) -I$(ROOT) $< -o $@
$(OUT)/ptx129.ptx: test/fixture_device.cu | check-cuda
	@mkdir -p $(OUT)
	$(NVCC) -ptx -arch=compute_61 -Wno-deprecated-gpu-targets -std=c++20 -ccbin $(CXX) -I$(ROOT) $< -o $@
$(OUT)/sass70.fatbin: test/fixture_device.cu | check-cuda
	@mkdir -p $(OUT)
	$(NVCC) -fatbin -gencode arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets -std=c++20 -ccbin $(CXX) -I$(ROOT) $< -o $@

$(DEVICE_FIXTURE): test/fixture_device.c $(OUT)/fixture_device.fatbin $(OUT)/fat129.fatbin $(OUT)/ptx129.ptx $(OUT)/sass70.fatbin eg_door.h eg_unit_meta.h
	$(call embed,$(OUT)/fixture_device.fatbin,fixture__image,$(OUT)/fixture_device_image.o)
	$(call embed,$(OUT)/fat129.fatbin,fat129__image,$(OUT)/fat129_image.o)
	$(call embed,$(OUT)/ptx129.ptx,ptx129__image,$(OUT)/ptx129_image.o)
	$(call embed,$(OUT)/sass70.fatbin,sass70__image,$(OUT)/sass70_image.o)
	$(CC) $(CFLAGS_C) -fPIC -shared -fvisibility=hidden -o $@ $< $(OUT)/fixture_device_image.o $(OUT)/fat129_image.o $(OUT)/ptx129_image.o $(OUT)/sass70_image.o

$(DEVICE_TEST): test/door_device_test.c eg_door.h $(LIB) | check-cuda
	$(CC) $(CFLAGS_C) -isystem $(CUDA)/include -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) $(CUDART) -lm -pthread

device: check-cuda symbols $(FIXTURE) $(DEVICE_FIXTURE) $(DEVICE_TEST)

PAIR_DEVICE_TEST := $(OUT)/pair_device_test
RUNNER_PAIR_TEST := $(OUT)/runner_pair_test
ACTIVE_PAIR_TEST := $(OUT)/active_pair_test
FEATURES_TEST := $(OUT)/device_features_test
FAST_PAIR_TEST := $(OUT)/fast_pair_test
GRAPH_PAIR_TEST := $(OUT)/graph_pair_test
OPS_TEST := $(OUT)/device_ops_test
BANDED_RUNNER_TEST := $(OUT)/banded_runner_test
DEVICE_REVERSE_TEST := $(OUT)/device_reverse_test

# The pair tests over test/pair_util.h: one rule, each binary built from the test/ source of its own name.
$(RUNNER_PAIR_TEST) $(ACTIVE_PAIR_TEST) $(FEATURES_TEST) $(OPS_TEST) $(BANDED_RUNNER_TEST) $(FAST_PAIR_TEST) $(GRAPH_PAIR_TEST) $(DEVICE_REVERSE_TEST): $(OUT)/%: test/%.c test/pair_util.h eg_door.h $(LIB) | check-cuda
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

pair-device-$(1): symbols $(OUT)/libpair_dev_$(1).so $$(PAIR_DEVICE_TEST) $$(RUNNER_PAIR_TEST) $$(ACTIVE_PAIR_TEST) $$(FAST_PAIR_TEST) $$(GRAPH_PAIR_TEST) $$(FEATURES_TEST) $$(OPS_TEST) $$(BANDED_RUNNER_TEST)
endef
$(foreach u,$(PAIR_DEVICE_UNITS),$(eval $(call PAIR_DEVICE_RULES,$(u))))

# pair-device-reverse: a recorded run and its reverse on the device, rr_pend and _timed: per unit one library of the old device
# run's own images (PTX_<unit>, PTX_<unit>_tape, PTX_<unit>_step_vjp: the PTX it loaded) with the host meta (META_reverse,
# META_timed) whose image fields name them (act: _pend_body, the active-set step: compacting every 4 and 16 steps, and reordering 4096 samples; auto: rr_pend_auto's K-step twin; kstep: steps_reverse FUSED); run $(OUT)/device_reverse_test <library> <oracle dir> <3 digests> <primal> <window>
# through the GPU pool against the old device run's states, report and gradients.
cuda_meta = sed 's/M(host_entry, "\([A-Za-z0-9_]*\)_host"), M(device_image, NULL), M(device_function, NULL)\(.*\)EG_IMAGE_NONE\(.*\)M(device_image_size, NULL)/M(host_entry, NULL), M(device_image, "\1__image"), M(device_function, "\1")\2EG_IMAGE_PTX\3M(device_image_size, "\1__image_size")/' $(1) > $(2)
define DEV_REV_RULES
$(OUT)/libpair_dev_rev_$(1).so: $$(PTX_$(1)) $$(PTX_$(1)_tape) $$(PTX_$(1)_step_vjp) $$(META_$(2)) eg_door.h eg_unit_meta.h | check-cuda
	@test -f "$$(PTX_$(1))" -a -f "$$(PTX_$(1)_tape)" -a -f "$$(PTX_$(1)_step_vjp)" || \
	  { echo "set PTX_$(1), PTX_$(1)_tape and PTX_$(1)_step_vjp"; exit 1; }
	@mkdir -p $(OUT)
	$$(call embed,$$(PTX_$(1)),$(3)__image,$(OUT)/dev_rev_$(1)_p.o)
	$$(call embed,$$(PTX_$(1)_tape),$(3)_tape__image,$(OUT)/dev_rev_$(1)_t.o)
	$$(call embed,$$(PTX_$(1)_step_vjp),$(3)_step_vjp__image,$(OUT)/dev_rev_$(1)_v.o)
	$$(call cuda_meta,$$(META_$(2)),$(OUT)/dev_rev_$(1)_meta.c)
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) -c $(OUT)/dev_rev_$(1)_meta.c -o $(OUT)/dev_rev_$(1)_meta.o
	$$(CC) -shared -o $$@ $(OUT)/dev_rev_$(1)_meta.o $(OUT)/dev_rev_$(1)_p.o $(OUT)/dev_rev_$(1)_t.o $(OUT)/dev_rev_$(1)_v.o
endef
$(eval $(call DEV_REV_RULES,rr_pend,reverse,rr_pend))
$(eval $(call DEV_REV_RULES,timed,timed,_timed))
$(eval $(call DEV_REV_RULES,act,act,_pend_body))
$(eval $(call DEV_REV_RULES,auto,auto,rr_pend_auto))
# kstep<K>: steps_reverse FUSED over the K-step primal rr_pend_x<K>: its twin and step vjp are rr_pend's, plus the K-step vjp.
define DEV_KSTEP_RULES
$(OUT)/libpair_dev_rev_$(1).so: $$(PTX_$(1)) $$(PTX_$(1)_tape) $$(PTX_$(1)_step_vjp) $$(PTX_$(1)_fused_vjp) $$(META_$(1)) eg_door.h eg_unit_meta.h | check-cuda
	@mkdir -p $(OUT)
	$$(call embed,$$(PTX_$(1)),$(2)__image,$(OUT)/dev_rev_$(1)_p.o)
	$$(call embed,$$(PTX_$(1)_tape),rr_pend_tape__image,$(OUT)/dev_rev_$(1)_t.o)
	$$(call embed,$$(PTX_$(1)_step_vjp),rr_pend_step_vjp__image,$(OUT)/dev_rev_$(1)_v.o)
	$$(call embed,$$(PTX_$(1)_fused_vjp),rr_pend_fused_vjp__image,$(OUT)/dev_rev_$(1)_f.o)
	$$(call cuda_meta,$$(META_$(1)),$(OUT)/dev_rev_$(1)_meta.c)
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) -c $(OUT)/dev_rev_$(1)_meta.c -o $(OUT)/dev_rev_$(1)_meta.o
	$$(CC) -shared -o $$@ $(OUT)/dev_rev_$(1)_meta.o $(OUT)/dev_rev_$(1)_p.o $(OUT)/dev_rev_$(1)_t.o $(OUT)/dev_rev_$(1)_v.o $(OUT)/dev_rev_$(1)_f.o
endef
$(eval $(call DEV_KSTEP_RULES,kstep,rr_pend_x4))
$(eval $(call DEV_KSTEP_RULES,kstep3,rr_pend_x3))
$(eval $(call DEV_KSTEP_RULES,kstep7,rr_pend_x7))
$(eval $(call DEV_KSTEP_RULES,kstep8,rr_pend_x8))
pair-device-reverse: symbols $(OUT)/libpair_dev_rev_rr_pend.so $(OUT)/libpair_dev_rev_timed.so $(OUT)/libpair_dev_rev_act.so $(OUT)/libpair_dev_rev_auto.so $(OUT)/libpair_dev_rev_kstep.so $(OUT)/libpair_dev_rev_kstep3.so $(OUT)/libpair_dev_rev_kstep7.so $(OUT)/libpair_dev_rev_kstep8.so $(DEVICE_REVERSE_TEST)

# pair-device-refine: events REFINE (a plain run) and eg_run_events on the device, per case one library of the old device run's own images
# (PTX_ref_<case>, _tape, _newton, _guard) with the host meta; run $(OUT)/device_refine_test <library> <case> <oracle dir> <4 digests> <max_steps>
# <on_no_event> <compare> through the GPU pool. Cases: osc, pend, grid (bitwise against the old device run), tangent, two (refused by name).
META_ref_osc ?= $(META_ev)
META_ref_pend ?= $(META_pend)
META_ref_grid ?= test/derived/ev_grid_meta.c
META_ref_tangent ?= test/derived/ev_tangent_meta.c
META_ref_two ?= test/derived/ev_two_meta.c
define DEV_REF_RULES
$(OUT)/libpair_dev_ref_$(1).so: $$(PTX_ref_$(1)) $$(PTX_ref_$(1)_tape) $$(PTX_ref_$(1)_newton) $$(PTX_ref_$(1)_guard) $$(META_ref_$(1)) eg_door.h eg_unit_meta.h | check-cuda
	@test -f "$$(PTX_ref_$(1))" -a -f "$$(PTX_ref_$(1)_tape)" -a -f "$$(PTX_ref_$(1)_newton)" -a -f "$$(PTX_ref_$(1)_guard)" || \
	  { echo "set PTX_ref_$(1), _tape, _newton and _guard"; exit 1; }
	@mkdir -p $(OUT)
	$$(call embed,$$(PTX_ref_$(1)),ev_$(1)__image,$(OUT)/dev_ref_$(1)_p.o)
	$$(call embed,$$(PTX_ref_$(1)_tape),ev_$(1)_tape__image,$(OUT)/dev_ref_$(1)_t.o)
	$$(call embed,$$(PTX_ref_$(1)_newton),ev_$(1)_newton__image,$(OUT)/dev_ref_$(1)_n.o)
	$$(call embed,$$(PTX_ref_$(1)_guard),ev_$(1)_state_guard_vjp__image,$(OUT)/dev_ref_$(1)_g.o)
	$$(call cuda_meta,$$(META_ref_$(1)),$(OUT)/dev_ref_$(1)_meta.c)
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) -c $(OUT)/dev_ref_$(1)_meta.c -o $(OUT)/dev_ref_$(1)_meta.o
	$$(CC) -shared -o $$@ $(OUT)/dev_ref_$(1)_meta.o $(OUT)/dev_ref_$(1)_p.o $(OUT)/dev_ref_$(1)_t.o $(OUT)/dev_ref_$(1)_n.o $(OUT)/dev_ref_$(1)_g.o
endef
$(foreach c,osc pend grid tangent two,$(eval $(call DEV_REF_RULES,$(c))))
DEVICE_REFINE_TEST := $(OUT)/device_refine_test
$(DEVICE_REFINE_TEST): test/device_refine_test.c test/pair_util.h eg_door.h $(LIB) | check-cuda
	$(CC) $(CFLAGS_C) -isystem $(CUDA)/include -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door \
	  -Wl,-rpath,$(OUT) $(CUDART) -lm -pthread
pair-device-refine: symbols $(foreach c,osc pend grid tangent two,$(OUT)/libpair_dev_ref_$(c).so) $(DEVICE_REFINE_TEST)

# pair-device-refine-reverse: a recorded events refine run (window 1) and eg_reverse through the stop on the device, per model one library of the
# old device run's own images (PTX_rrf_<model>, _tape, _step_vjp, _newton, _guard, _refine_vjp) with the host meta; run
# $(OUT)/device_refine_reverse_test <library> <oracle dir> <6 digests: primal, twin, Newton, guard, step vjp, refine vjp> [pend] through the GPU pool.
define DEV_RRF_RULES
$(OUT)/libpair_dev_rrf_$(1).so: $$(PTX_rrf_$(1)) $$(PTX_rrf_$(1)_tape) $$(PTX_rrf_$(1)_step_vjp) $$(PTX_rrf_$(1)_newton) $$(PTX_rrf_$(1)_guard) $$(PTX_rrf_$(1)_refine_vjp) $$(META_rrf_$(1)) eg_door.h eg_unit_meta.h | check-cuda
	@mkdir -p $(OUT)
	$$(call embed,$$(PTX_rrf_$(1)),ev_$(1)__image,$(OUT)/dev_rrf_$(1)_p.o)
	$$(call embed,$$(PTX_rrf_$(1)_tape),ev_$(1)_tape__image,$(OUT)/dev_rrf_$(1)_t.o)
	$$(call embed,$$(PTX_rrf_$(1)_step_vjp),ev_$(1)_step_vjp__image,$(OUT)/dev_rrf_$(1)_s.o)
	$$(call embed,$$(PTX_rrf_$(1)_newton),ev_$(1)_newton__image,$(OUT)/dev_rrf_$(1)_n.o)
	$$(call embed,$$(PTX_rrf_$(1)_guard),ev_$(1)_state_guard_vjp__image,$(OUT)/dev_rrf_$(1)_g.o)
	$$(call embed,$$(PTX_rrf_$(1)_refine_vjp),ev_$(1)_refine_vjp__image,$(OUT)/dev_rrf_$(1)_r.o)
	$$(call cuda_meta,$$(META_rrf_$(1)),$(OUT)/dev_rrf_$(1)_meta.c)
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) -c $(OUT)/dev_rrf_$(1)_meta.c -o $(OUT)/dev_rrf_$(1)_meta.o
	$$(CC) -shared -o $$@ $(OUT)/dev_rrf_$(1)_meta.o $(OUT)/dev_rrf_$(1)_p.o $(OUT)/dev_rrf_$(1)_t.o $(OUT)/dev_rrf_$(1)_s.o $(OUT)/dev_rrf_$(1)_n.o $(OUT)/dev_rrf_$(1)_g.o $(OUT)/dev_rrf_$(1)_r.o
endef
META_rrf_osc ?= $(META_ref_osc)
META_rrf_pend ?= $(META_ref_pend)
META_rrf_grid ?= test/derived/ev_grid_rr_meta.c
$(foreach c,osc pend grid,$(eval $(call DEV_RRF_RULES,$(c))))
DEVICE_RRF_TEST := $(OUT)/device_refine_reverse_test
$(DEVICE_RRF_TEST): test/device_refine_reverse_test.c test/pair_util.h eg_door.h $(LIB) | check-cuda
	$(CC) $(CFLAGS_C) -isystem $(CUDA)/include -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door \
	  -Wl,-rpath,$(OUT) $(CUDART) -lm -pthread
pair-device-refine-reverse: symbols $(foreach c,osc pend grid,$(OUT)/libpair_dev_rrf_$(c).so) $(DEVICE_RRF_TEST)

# pair-refine-pend, pair-refine-reverse-pend: the pendulum (ev_pend, the old path's second event model, a 0.05, c 0.2) over its own
# library (META_pend) with the same two tests; ORACLE_pend (plain refine run), ORACLE_pendrr (the reverse's gradients).
META_pend ?= test/derived/ev_pend_meta.c
DIGEST_pend ?= 95c987555a3e3adbd7bd74fbda8a114054889e69547e12adfef3224f77a49eae
DIGEST_pend_tape ?= f501f4c247a820fb7df44bbcb23acadc92c6e7afca6346a24a2a9fa6a126063e
DIGEST_pend_step_vjp ?= a6e50ecab8d027681391468b32768ada530ca95a073a4ddf78d5ec6c2cc4d0a3
DIGEST_pend_newton ?= 7fa4b726b43f41b19d9bebf8d812b371eb6e4f3a43426f4fe9ad3c69efbbbc25
DIGEST_pend_guard ?= 012d7e9df31a4822327f8c88a45d0bfdbe5f8f3dfba0871c0edd02ff073028bc
DIGEST_pend_refine_vjp ?= d6c8c1c8e308deccbe8f6e8207b0bcc08abbe26730dda782fb75442bfa197588
$(OUT)/libpair_refine_pend.so: $(KERNEL_pend) $(KERNEL_pend_tape) $(KERNEL_pend_step_vjp) $(KERNEL_pend_newton) $(KERNEL_pend_guard) $(KERNEL_pend_refine_vjp) $(META_pend) eg_door.h eg_unit_meta.h | check-inputs
	@test -f "$(KERNEL_pend)" -a -f "$(KERNEL_pend_tape)" -a -f "$(KERNEL_pend_step_vjp)" -a -f "$(KERNEL_pend_newton)" -a -f "$(KERNEL_pend_guard)" -a -f "$(KERNEL_pend_refine_vjp)" -a -f "$(META_pend)" -a \
	  -d "$(PAIR_WING_DIR)" || { echo "set KERNEL_pend, _tape, _step_vjp, _newton, _guard, _refine_vjp, META_pend and PAIR_WING_DIR"; exit 1; }
	@mkdir -p $(OUT)
	$(CXX) $(PAIR_HOST_FLAGS) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_pend) -o $(OUT)/pend_primal.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,tape) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_pend_tape) -o $(OUT)/pend_tape.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,stepvjp) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_pend_step_vjp) -o $(OUT)/pend_stepvjp.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,newton) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_pend_newton) -o $(OUT)/pend_newton.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,guard) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_pend_guard) -o $(OUT)/pend_guard.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,refvjp) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_pend_refine_vjp) -o $(OUT)/pend_refvjp.o
	$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$(ROOT) -c $(META_pend) -o $(OUT)/pend_meta.o
	$(CXX) -shared -o $@ $(OUT)/pend_primal.o $(OUT)/pend_tape.o $(OUT)/pend_stepvjp.o $(OUT)/pend_newton.o $(OUT)/pend_guard.o $(OUT)/pend_refvjp.o $(OUT)/pend_meta.o

pair-refine-pend: symbols $(OUT)/libpair_refine_pend.so $(REFINE_TEST)
	$(REFINE_TEST) $(OUT)/libpair_refine_pend.so $(ORACLE_pend) $(DIGEST_pend) $(DIGEST_pend_tape) $(DIGEST_pend_newton) $(DIGEST_pend_guard) 400 0 1 pend

pair-refine-reverse-pend: symbols $(OUT)/libpair_refine_pend.so $(REFINE_REVERSE_TEST)
	$(REFINE_REVERSE_TEST) $(OUT)/libpair_refine_pend.so $(ORACLE_pendrr) $(DIGEST_pend) $(DIGEST_pend_tape) $(DIGEST_pend_newton) $(DIGEST_pend_guard) \
	  $(DIGEST_pend_step_vjp) $(DIGEST_pend_refine_vjp) pend

# pair-refine-reverse-grid: ev_grid (a horizon on the step grid, per-sample dt and big_t, every sample stops AT_STEP) recorded with its taping
# twin, step vjp, Newton, state guard vjp and refine vjp (library META_rgrid; kernels KERNEL_rgrid, _tape, _step_vjp, _newton, _guard,
# _refine_vjp); the end states, stop times, statuses and the gradients under seeds t and x equal the old path's (ORACLE_rgrid).
META_rgrid ?= test/derived/ev_grid_rr_meta.c
DIGEST_rgrid ?= 015887e341d24a67e9b6cdbb426a7c5e6f2236a89272b671b4c9441f57694875
DIGEST_rgrid_tape ?= b944fc501b4889d02b1defee7fbbd294dce5ca35587fbba179c47a098b36a5e0
DIGEST_rgrid_step_vjp ?= 4d865ef658bbac28398db5811d9247a02ef5ea5facbbd7d314af131452c158ea
DIGEST_rgrid_newton ?= 6eb1179bb45cf34803a9a0a8f89776af246577ab76769a1efc4cae1ee72cc8dc
DIGEST_rgrid_guard ?= 8e5d6669ba092c5009dafaa13dd2fc3fb8a11d4259d6f83d85594550d9ad34f7
DIGEST_rgrid_refine_vjp ?= 04e2e466da275c57c6e96d7995430406e0cfc9f315a2cd31d2b3a5c51866be57
$(OUT)/libpair_refine_grid.so: $(KERNEL_rgrid) $(KERNEL_rgrid_tape) $(KERNEL_rgrid_step_vjp) $(KERNEL_rgrid_newton) $(KERNEL_rgrid_guard) $(KERNEL_rgrid_refine_vjp) $(META_rgrid) eg_door.h eg_unit_meta.h | check-inputs
	@test -f "$(KERNEL_rgrid)" -a -f "$(KERNEL_rgrid_tape)" -a -f "$(KERNEL_rgrid_step_vjp)" -a -f "$(KERNEL_rgrid_newton)" -a -f "$(KERNEL_rgrid_guard)" -a -f "$(KERNEL_rgrid_refine_vjp)" -a -f "$(META_rgrid)" -a \
	  -d "$(PAIR_WING_DIR)" || { echo "set KERNEL_rgrid, _tape, _step_vjp, _newton, _guard, _refine_vjp, META_rgrid and PAIR_WING_DIR"; exit 1; }
	@mkdir -p $(OUT)
	$(CXX) $(PAIR_HOST_FLAGS) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_rgrid) -o $(OUT)/rgrid_primal.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,tape) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_rgrid_tape) -o $(OUT)/rgrid_tape.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,stepvjp) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_rgrid_step_vjp) -o $(OUT)/rgrid_stepvjp.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,newton) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_rgrid_newton) -o $(OUT)/rgrid_newton.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,guard) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_rgrid_guard) -o $(OUT)/rgrid_guard.o
	$(CXX) $(PAIR_HOST_FLAGS) $(call REV_RENAME,refvjp) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) -c $(KERNEL_rgrid_refine_vjp) -o $(OUT)/rgrid_refvjp.o
	$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$(ROOT) -c $(META_rgrid) -o $(OUT)/rgrid_meta.o
	$(CXX) -shared -o $@ $(OUT)/rgrid_primal.o $(OUT)/rgrid_tape.o $(OUT)/rgrid_stepvjp.o $(OUT)/rgrid_newton.o $(OUT)/rgrid_guard.o $(OUT)/rgrid_refvjp.o $(OUT)/rgrid_meta.o

pair-refine-reverse-grid: symbols $(OUT)/libpair_refine_grid.so $(REFINE_REVERSE_TEST)
	$(REFINE_REVERSE_TEST) $(OUT)/libpair_refine_grid.so $(ORACLE_rgrid) $(DIGEST_rgrid) $(DIGEST_rgrid_tape) $(DIGEST_rgrid_newton) $(DIGEST_rgrid_guard) \
	  $(DIGEST_rgrid_step_vjp) $(DIGEST_rgrid_refine_vjp) grid


# pair-refine-reverse-grid-refused: the same units with a per-sample read's gradient declared reduced 1 (on the step vjp, META_rgrid_bad_step;
# on the refine vjp, META_rgrid_bad_refine): the runner is refused BAD_REQUEST naming "reduced".
META_rgrid_bad_step ?= test/derived/ev_grid_rr_bad_step_meta.c
META_rgrid_bad_refine ?= test/derived/ev_grid_rr_bad_refine_meta.c
define GRID_BAD_RULES
$(OUT)/libpair_refine_grid_bad_$(1).so: $$(KERNEL_rgrid) $$(KERNEL_rgrid_tape) $$(KERNEL_rgrid_step_vjp) $$(KERNEL_rgrid_newton) $$(KERNEL_rgrid_guard) $$(KERNEL_rgrid_refine_vjp) $$(META_rgrid_bad_$(1)) eg_door.h eg_unit_meta.h | check-inputs
	@mkdir -p $(OUT)
	$$(CXX) $$(PAIR_HOST_FLAGS) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_rgrid) -o $(OUT)/rgb_$(1)_primal.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,tape) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_rgrid_tape) -o $(OUT)/rgb_$(1)_tape.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,stepvjp) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_rgrid_step_vjp) -o $(OUT)/rgb_$(1)_stepvjp.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,newton) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_rgrid_newton) -o $(OUT)/rgb_$(1)_newton.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,guard) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_rgrid_guard) -o $(OUT)/rgb_$(1)_guard.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,refvjp) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_rgrid_refine_vjp) -o $(OUT)/rgb_$(1)_refvjp.o
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) -c $$(META_rgrid_bad_$(1)) -o $(OUT)/rgb_$(1)_meta.o
	$$(CXX) -shared -o $$@ $(OUT)/rgb_$(1)_primal.o $(OUT)/rgb_$(1)_tape.o $(OUT)/rgb_$(1)_stepvjp.o $(OUT)/rgb_$(1)_newton.o $(OUT)/rgb_$(1)_guard.o $(OUT)/rgb_$(1)_refvjp.o $(OUT)/rgb_$(1)_meta.o
endef
$(eval $(call GRID_BAD_RULES,step))
$(eval $(call GRID_BAD_RULES,refine))

pair-refine-reverse-grid-refused: symbols $(OUT)/libpair_refine_grid_bad_step.so $(OUT)/libpair_refine_grid_bad_refine.so $(REFINE_REVERSE_TEST)
	$(REFINE_REVERSE_TEST) $(OUT)/libpair_refine_grid_bad_step.so $(ORACLE_rgrid) $(DIGEST_rgrid) $(DIGEST_rgrid_tape) $(DIGEST_rgrid_newton) $(DIGEST_rgrid_guard) \
	  $(DIGEST_rgrid_step_vjp) $(DIGEST_rgrid_refine_vjp) grid-refused
	$(REFINE_REVERSE_TEST) $(OUT)/libpair_refine_grid_bad_refine.so $(ORACLE_rgrid) $(DIGEST_rgrid) $(DIGEST_rgrid_tape) $(DIGEST_rgrid_newton) $(DIGEST_rgrid_guard) \
	  $(DIGEST_rgrid_step_vjp) $(DIGEST_rgrid_refine_vjp) grid-refused

# pair-status-<case>: the refine statuses beyond REFINED over one library per case (META test/derived/ev_<case>_meta.c, kernels
# KERNEL_s<case>, _tape, _newton, _guard): ev_grid AT_STEP (bitwise against ORACLE_sgrid), ev_tangent GRAZE and ev_two TWO_RULES (refused by name).
STATUS_TEST := $(OUT)/refine_status_test
$(STATUS_TEST): test/refine_status_test.c test/pair_util.h eg_door.h $(LIB)
	$(CC) $(CFLAGS_C) -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) -lm -pthread
define STATUS_RULES
$(OUT)/libpair_status_$(1).so: $$(KERNEL_s$(1)) $$(KERNEL_s$(1)_tape) $$(KERNEL_s$(1)_newton) $$(KERNEL_s$(1)_guard) test/derived/ev_$(1)_meta.c eg_door.h eg_unit_meta.h | check-inputs
	@mkdir -p $$(OUT)
	$$(CXX) $$(PAIR_HOST_FLAGS) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_s$(1)) -o $$(OUT)/st_$(1)_primal.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,tape) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_s$(1)_tape) -o $$(OUT)/st_$(1)_tape.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,newton) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_s$(1)_newton) -o $$(OUT)/st_$(1)_newton.o
	$$(CXX) $$(PAIR_HOST_FLAGS) $$(call REV_RENAME,guard) -I$$(AETHER_INCLUDE) -I$$(AETHER_INCLUDE)/eagle -I$$(PAIR_WING_DIR) -c $$(KERNEL_s$(1)_guard) -o $$(OUT)/st_$(1)_guard.o
	$$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$$(ROOT) -c test/derived/ev_$(1)_meta.c -o $$(OUT)/st_$(1)_meta.o
	$$(CXX) -shared -o $$@ $$(OUT)/st_$(1)_primal.o $$(OUT)/st_$(1)_tape.o $$(OUT)/st_$(1)_newton.o $$(OUT)/st_$(1)_guard.o $$(OUT)/st_$(1)_meta.o
pair-status-$(1): symbols $(OUT)/libpair_status_$(1).so $(STATUS_TEST)
	$(STATUS_TEST) $(OUT)/libpair_status_$(1).so $(1) $(or $(ORACLE_s$(1)),-) $(DIGEST_s$(1)) $(DIGEST_s$(1)_tape) $(DIGEST_s$(1)_newton) $(DIGEST_s$(1)_guard)
endef
$(eval $(call STATUS_RULES,grid))
$(eval $(call STATUS_RULES,tangent))
$(eval $(call STATUS_RULES,two))
