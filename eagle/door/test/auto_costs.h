/* SPDX-License-Identifier: Apache-2.0 */

/* The branch costs of the AUTO units of the fixture libraries (test/fixture_unit.c, test/fixture_device.c), one source for them and for the
 * recorder that asks the old stack what it decides on them (test/auto_record.py parses this file). Each row is
 *   flags, arith32, arith64, special32, special64, loads, gathers, chain32, chain64, chain_special32, chain_special64, stores
 * with EG_BRANCH_* flags; each unit's dispatch cost_flags are AUTO_FLAGS_<name>. "rare" prices a cheap branch beside a heavy one, "even" two equal
 * heavy branches under a vectorised fused loop, "scalar" two cheap branches under a scalar fused loop (splitting never pays). */
#define AUTO_FLAGS_rare 3u
#define AUTO_COST_rare_0 5u, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 2
#define AUTO_COST_rare_1 5u, 0, 40, 0, 20, 2, 0, 0, 0, 0, 0, 2
#define AUTO_FLAGS_even 3u
#define AUTO_COST_even_0 5u, 0, 40, 0, 20, 2, 0, 0, 0, 0, 0, 2
#define AUTO_COST_even_1 5u, 0, 40, 0, 20, 2, 0, 0, 0, 0, 0, 2
#define AUTO_FLAGS_scalar 2u
#define AUTO_COST_scalar_0 4u, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 2
#define AUTO_COST_scalar_1 4u, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 2
