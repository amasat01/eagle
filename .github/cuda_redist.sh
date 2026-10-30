#!/bin/sh
# The CUDA 12.9 toolchain (nvcc, the runtime headers with the driver stub, CCCL) from NVIDIA's versioned redist archives: the one place that
# pins each archive's version and sha256, per machine (aarch64 is NVIDIA's `sbsa`). A download is kept in a cache
# directory, which CI keys by these sums; every archive is verified before it is unpacked, cached or fresh.
#   cuda_redist.sh key   <x86_64|aarch64>                    the cache key: the pinned versions and sums
#   cuda_redist.sh fetch <x86_64|aarch64> <prefix> <cache>   unpack the archives under <prefix>, then run its nvcc
set -eu

cmd=${1:?key|fetch}
case "${2:?machine}" in
  x86_64)  arch=x86_64
           parts="cuda_nvcc 12.9.86 7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35
                  cuda_cudart 12.9.79 1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99
                  cuda_cccl 12.9.27 8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa" ;;
  aarch64) arch=sbsa
           parts="cuda_nvcc 12.9.86 0aa1fce92dbae76c059c27eefb9d0ffb58e1291151e44ff7c7f1fc2dd9376c0d
                  cuda_cudart 12.9.79 8b422a3b2cb8452cb678181b0bf9d7aa7342df168b5319c5488ae3b8514101fc
                  cuda_cccl 12.9.27 8c3da24801b500f1d9217d191bb4b63e5d2096c8e7d0b7695e876853180ba82f" ;;
  *) echo "cuda_redist: no CUDA redist pins for '$2'" >&2; exit 2 ;;
esac

if [ "$cmd" != key ]; then prefix=${3:?prefix}; cache=${4:?cache dir}; fi
# shellcheck disable=SC2086
set -- $parts
if [ "$cmd" = key ]; then echo "$*" | tr ' ' -; exit 0; fi

mkdir -p "$prefix" "$cache"
while [ $# -gt 0 ]; do
  name=$1; version=$2; sum=$3; shift 3
  archive="$name-linux-$arch-$version-archive.tar.xz"
  if ! { [ -f "$cache/$archive" ] && echo "$sum  $cache/$archive" | sha256sum -c - >/dev/null 2>&1; }; then
    curl -fsSL --retry 5 -o "$cache/$archive" "https://developer.download.nvidia.com/compute/cuda/redist/$name/linux-$arch/$archive"
  fi
  echo "$sum  $cache/$archive" | sha256sum -c -
  tar -xJf "$cache/$archive" -C "$prefix" --strip-components=1
done
"$prefix/bin/nvcc" --version
