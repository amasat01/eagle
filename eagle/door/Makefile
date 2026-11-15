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
# and run the C test), all (= test), pair (see below).
#
#   make ... PAIR_KERNEL=<kernel .cpp> PAIR_META=<meta .c> PAIR_WING_DIR=<raptor wing dir> \
#            PAIR_ORACLE=<dir with the old path's outputs> pair
#
# pair: one generated unit (its kernel text and its generated meta source)
# built into one unit library and run through the door against the outputs the
# old path wrote for the same case (y_scalar_a.f64, y_plane_a.f64).

CXX ?= g++
CC ?= gcc
OUT ?= $(CURDIR)/build
AETHER_INCLUDE ?=
WING_FACTS ?=

PAIR_KERNEL ?=
PAIR_META ?=
PAIR_WING_DIR ?=
PAIR_ORACLE ?=
# The unit the caller names (name + digest) and the host entry its meta names.
PAIR_UNIT ?= ss_bcast
PAIR_DIGEST ?= 49a12f72dfd7520aa76ed07e9c4765ed5f4c5a6d4d2b29b5ad19eb91767befec
PAIR_ENTRY ?= ss_bcast_host
# The installed host unit recipe (the host compiler flags the old path builds a unit with).
PAIR_HOST_FLAGS ?= -O3 -ffp-contract=fast -fno-math-errno -march=native -mprefer-vector-width=512 \
  -fno-trapping-math -DAETHER_HOST_VECTOR_MATH -DAETHER_HVM_MASKED=1 -std=c++23 -fPIC -DAETHER_CPP_MODE

ROOT := $(abspath $(CURDIR)/../..)
WARN := -Wall -Wextra -pedantic -Werror
CXXFLAGS_DOOR := -std=c++23 -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
  -ffp-contract=off $(WARN) -DAETHER_CPP_MODE=1 -DEAGLE_CPU_ONLY=1 \
  -include $(WING_FACTS) -I$(ROOT) -I$(AETHER_INCLUDE)
CFLAGS_C := -std=c11 -O2 -ffp-contract=off $(WARN) -I$(ROOT)

SRC := Context.cpp Unit.cpp Plane.cpp Plan.cpp abi.cpp
OBJ := $(SRC:%.cpp=$(OUT)/%.o)
LIB := $(OUT)/libeagle_door.so
FIXTURE := $(OUT)/libfixture_unit.so
TEST := $(OUT)/door_test

.PHONY: all lib symbols test check-inputs pair
all: test

check-inputs:
	@test -n "$(AETHER_INCLUDE)" || { echo "set AETHER_INCLUDE"; exit 1; }
	@test -f "$(WING_FACTS)" || { echo "set WING_FACTS to the wing's raptor_facts.h"; exit 1; }

$(OUT)/%.o: %.cpp Context.h eg_door.h eg_unit_meta.h | check-inputs
	@mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS_DOOR) -c $< -o $@

$(LIB): $(OBJ) door.map
	$(CXX) -shared -o $@ $(OBJ) -Wl,--version-script=$(CURDIR)/door.map -Wl,--no-undefined -pthread -ldl

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

PAIR_LIB := $(OUT)/libpair_unit.so
PAIR_TEST := $(OUT)/pair_test

# The kernel text as the old path compiles it; the meta as C11, pedantic and warning-free.
$(PAIR_LIB): $(PAIR_KERNEL) $(PAIR_META) eg_door.h eg_unit_meta.h | check-inputs
	@test -f "$(PAIR_KERNEL)" -a -f "$(PAIR_META)" -a -d "$(PAIR_WING_DIR)" || \
	  { echo "set PAIR_KERNEL, PAIR_META and PAIR_WING_DIR"; exit 1; }
	@mkdir -p $(OUT)
	$(CXX) $(PAIR_HOST_FLAGS) -I$(AETHER_INCLUDE) -I$(AETHER_INCLUDE)/eagle -I$(PAIR_WING_DIR) \
	  -c $(PAIR_KERNEL) -o $(OUT)/pair_kernel.o
	$(CC) -std=c11 -Wall -Wextra -pedantic -Werror -fPIC -fvisibility=hidden -I$(CURDIR) \
	  -c $(PAIR_META) -o $(OUT)/pair_meta.o
	$(CXX) -shared -o $@ $(OUT)/pair_kernel.o $(OUT)/pair_meta.o

$(PAIR_TEST): test/pair_test.c eg_door.h $(LIB)
	$(CC) $(CFLAGS_C) -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) -pthread

pair: symbols $(PAIR_LIB) $(PAIR_TEST)
	@for sym in eg_unit_library_meta $(PAIR_ENTRY); do \
	  nm -D --defined-only $(PAIR_LIB) | awk '{print $$3}' | grep -qx "$$sym" || \
	    { echo "FAIL  $(PAIR_LIB) does not export $$sym"; exit 1; }; \
	  echo "PASS  $(PAIR_LIB) exports $$sym"; done
	$(PAIR_TEST) $(PAIR_LIB) $(PAIR_ORACLE) $(PAIR_UNIT) $(PAIR_DIGEST)
