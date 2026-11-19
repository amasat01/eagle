#!/bin/bash
# eagle's host loops on its host pool against the same loops under OpenMP.
#
#   run.sh <new tree> <old tree> <aether include dir> <raptor_facts.h> <work dir> [rounds]
#   BENCH=bench_filtering OPS="scan compact reorder" run.sh ...      the filtering operations instead of the reduction
#
# <new tree> is this tree; <old tree> a checkout of the tree before the loops moved to the pool (its headers are the
# include root of the OpenMP build: no loop is copied). Both builds use the one source (bench.cpp, or bench_filtering.cpp
# for BENCH=bench_filtering). The two binaries run interleaved, round after round, on cores 2-7; each cell is the median
# and the interquartile range of the rounds' medians (microseconds). Teams are 1, 4 and every core of the set.
set -eu
new=$1 old=$2 aether=$3 facts=$4 work=$5 rounds=${6:-15}
bench=${BENCH:-bench}
ops=${OPS:-reduce}
src=$(dirname "$(readlink -f "$0")")/$bench.cpp
mkdir -p "$work"
flags="-std=c++23 -O2 -ffp-contract=off -DAETHER_CPP_MODE=1 -DEAGLE_CPU_ONLY=1 -include $facts -I$aether"
"${CXX:-g++}" $flags -I"$new" -pthread -o "$work/${bench}_pool" "$src"
"${CXX:-g++}" $flags -I"$old" -fopenmp -DEAGLE_BENCH_OMP -o "$work/${bench}_omp" "$src"
cores=2-7
all=6
echo "op team n pool_median pool_iqr omp_median omp_iqr bits_equal"
for op in $ops; do
  arg=$op
  [ "$bench" = bench ] && arg=
  for n in 10000 1000000 10000000; do
    for team in 1 4 $all; do
      : > "$work/pool.txt"; : > "$work/omp.txt"
      for r in $(seq "$rounds"); do
        taskset -c $cores "$work/${bench}_pool" $arg "$team" "$n" >> "$work/pool.txt"
        OMP_NUM_THREADS=$team taskset -c $cores "$work/${bench}_omp" $arg "$team" "$n" >> "$work/omp.txt"
      done
      python3 - "$op" "$team" "$n" "$work/pool.txt" "$work/omp.txt" <<'E'
import statistics, sys
op, team, n, pool, omp = sys.argv[1:6]
def read(path):
    rows = [line.split() for line in open(path)]
    return [float(a) for a, _ in rows], {b for _, b in rows}
def cell(xs):
    q = statistics.quantiles(xs, n=4)
    return f"{statistics.median(xs):.1f} {q[2] - q[0]:.1f}"
p, pb = read(pool)
o, ob = read(omp)
print(op, team, n, cell(p), cell(o), pb == ob and len(pb) == 1)
E
    done
  done
done
