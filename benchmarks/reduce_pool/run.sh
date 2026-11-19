#!/bin/bash
# The canonical host reduction on eagle's host pool against the same reduction under OpenMP.
#
#   run.sh <new tree> <old tree> <aether include dir> <raptor_facts.h> <work dir> [rounds]
#
# <new tree> is this tree; <old tree> a checkout of the tree before the loops moved to the pool (its headers are the
# include root of the OpenMP build: no loop is copied). Both builds use bench.cpp. The two binaries run interleaved,
# round after round, on cores 2-7; each cell is the median and the interquartile range of the rounds' medians
# (microseconds). Teams are 1, 4 and every core of the set.
set -eu
new=$1 old=$2 aether=$3 facts=$4 work=$5 rounds=${6:-15}
src=$(dirname "$(readlink -f "$0")")/bench.cpp
mkdir -p "$work"
flags="-std=c++23 -O2 -ffp-contract=off -DAETHER_CPP_MODE=1 -DEAGLE_CPU_ONLY=1 -include $facts -I$aether"
"${CXX:-g++}" $flags -I"$new" -pthread -o "$work/bench_pool" "$src"
"${CXX:-g++}" $flags -I"$old" -fopenmp -DEAGLE_BENCH_OMP -o "$work/bench_omp" "$src"
cores=2-7
all=6
echo "team n pool_median pool_iqr omp_median omp_iqr bits_equal"
for n in 10000 10000000; do
  for team in 1 4 $all; do
    : > "$work/pool.txt"; : > "$work/omp.txt"
    for r in $(seq "$rounds"); do
      taskset -c $cores "$work/bench_pool" "$team" "$n" >> "$work/pool.txt"
      OMP_NUM_THREADS=$team taskset -c $cores "$work/bench_omp" "$team" "$n" >> "$work/omp.txt"
    done
    python3 - "$team" "$n" "$work/pool.txt" "$work/omp.txt" <<'E'
import statistics, sys
team, n, pool, omp = sys.argv[1:5]
def read(path):
    rows = [line.split() for line in open(path)]
    return [float(a) for a, _ in rows], {b for _, b in rows}
def cell(xs):
    q = statistics.quantiles(xs, n=4)
    return f"{statistics.median(xs):.1f} {q[2] - q[0]:.1f}"
p, pb = read(pool)
o, ob = read(omp)
print(team, n, cell(p), cell(o), pb == ob and len(pb) == 1)
E
  done
done
