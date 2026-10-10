#!/usr/bin/env bash
# Build a free-threaded CPython with ThreadSanitizer, the interpreter the
# tsan-ft gate (tsan_ft.sh) runs eagle._core under.
#
#   tests/sanitize/build_tsan_python.sh <version> <prefix>
#
# CC/CXX default to clang/clang++ (CPython's --with-thread-sanitizer is
# clang-first); JOBS defaults to nproc. The result is <prefix>/bin/python3.Xt
# with the GIL off by default, plus CPython's own free-threading TSan
# suppressions copied to <prefix>/share/tsan/suppressions_free_threading.txt.
# Idempotent: an existing prefix whose interpreter reports the same version
# and TSan configure flags is kept as is (CI caches the prefix).
set -euo pipefail

version="${1:?usage: build_tsan_python.sh <version> <prefix>}"
prefix="${2:?usage: build_tsan_python.sh <version> <prefix>}"
minor="${version%.*}"
py="${prefix}/bin/python${minor}t"

if [ -x "$py" ] && "$py" -c "
import sys, sysconfig
args = sysconfig.get_config_var('CONFIG_ARGS') or ''
sys.exit(0 if sys.version.split()[0] == '${version}' and '--with-thread-sanitizer' in args
         and '--disable-gil' in args else 1)"; then
    echo "TSan CPython ${version} already at ${prefix}"
    exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
curl -fsSL -o "$work/py.tgz" "https://www.python.org/ftp/python/${version}/Python-${version}.tgz"
tar xzf "$work/py.tgz" -C "$work"
src="$work/Python-${version}"
(
    cd "$src"
    CC="${CC:-clang}" CXX="${CXX:-clang++}" ./configure --disable-gil --with-thread-sanitizer \
        --without-ensurepip --prefix="$prefix" > "$work/configure.log" 2>&1 \
        || { tail -40 "$work/configure.log"; exit 1; }
    make -j"${JOBS:-$(nproc)}" > "$work/make.log" 2>&1 || { tail -40 "$work/make.log"; exit 1; }
    make install > "$work/install.log" 2>&1 || { tail -40 "$work/install.log"; exit 1; }
)
mkdir -p "$prefix/share/tsan"
cp "$src/Tools/tsan/suppressions_free_threading.txt" "$prefix/share/tsan/"
"$py" -c "import sys; assert not sys._is_gil_enabled(), 'GIL is on'; print(sys.version)"
