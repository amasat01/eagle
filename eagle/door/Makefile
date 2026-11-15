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
# and run the C test), all (= test).

CXX ?= g++
CC ?= gcc
OUT ?= $(CURDIR)/build
AETHER_INCLUDE ?=
WING_FACTS ?=

ROOT := $(abspath $(CURDIR)/../..)
WARN := -Wall -Wextra -Werror
CXXFLAGS_DOOR := -std=c++23 -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
  -ffp-contract=off $(WARN) -DAETHER_CPP_MODE=1 -DEAGLE_CPU_ONLY=1 \
  -include $(WING_FACTS) -I$(ROOT) -I$(AETHER_INCLUDE)
CFLAGS_C := -std=c11 -O2 -ffp-contract=off $(WARN) -I$(ROOT)

SRC := Context.cpp Unit.cpp Plane.cpp Plan.cpp abi.cpp
OBJ := $(SRC:%.cpp=$(OUT)/%.o)
LIB := $(OUT)/libeagle_door.so
FIXTURE := $(OUT)/libfixture_unit.so
TEST := $(OUT)/door_test

.PHONY: all lib symbols test check-inputs
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

$(FIXTURE): test/fixture_unit.c eg_door.h eg_unit_meta.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS_C) -fPIC -shared -fvisibility=hidden -o $@ $<

$(TEST): test/door_test.c eg_door.h $(LIB)
	$(CC) $(CFLAGS_C) -Itest/c_shim -o $@ $< -L$(OUT) -leagle_door -Wl,-rpath,$(OUT) -pthread

test: symbols $(FIXTURE) $(TEST)
	$(TEST) $(FIXTURE)
