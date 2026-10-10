#!/usr/bin/env bash
# tsan-ft gate: eagle._core built with ThreadSanitizer, the free-threading
# (`ft`) rows run under a TSan free-threaded CPython (build_tsan_python.sh).
#
#   tests/sanitize/tsan_ft.sh <tsan-python>        (run from the eagle root)
#
# Inputs from the environment: CMAKE_PREFIX_PATH (aether + eagle headers, as
# for any binding build), RAPTOR_DIR (a raptor checkout; else raptor-core from
# PyPI), CC/CXX (default clang/clang++; TSan runtimes must match the
# interpreter's). The CUDA plugin is not built: the ft rows that need a GPU
# skip, as on a GPU-less runner.
#
# GREEN needs all of:
#   (i)   the interpreter is TSan-built with the GIL off, and _core carries TSan
#         instrumentation;
#   (ii)  the canary race IS reported without eagle's suppressions (the
#         detector is live and reaches _core);
#   (iii) the ft rows collect, pass, and produce ZERO TSan reports with
#         CPython's suppressions + tests/sanitize/tsan.supp.
set -euo pipefail

py="${1:?usage: tests/sanitize/tsan_ft.sh <tsan-python>}"
root="$(pwd)"
[ -f "$root/python/pyproject.toml" ] || { echo "run from the eagle root"; exit 2; }
work="$root/build/tsan"
prefix="$(cd "$(dirname "$py")/.." && pwd)"
cpython_supp="$prefix/share/tsan/suppressions_free_threading.txt"
[ -f "$cpython_supp" ] || { echo "RED: no CPython TSan suppressions at $cpython_supp"; exit 1; }
mkdir -p "$work"
cat "$cpython_supp" "$root/tests/sanitize/tsan.supp" > "$work/all.supp"
export CC="${CC:-clang}" CXX="${CXX:-clang++}" CUDA_VISIBLE_DEVICES=""

# (i) the interpreter
"$py" -c "
import sys, sysconfig
assert '--with-thread-sanitizer' in (sysconfig.get_config_var('CONFIG_ARGS') or ''), 'not a TSan build'
assert sys._is_gil_enabled() is False, 'GIL is enabled'
print('TSan CPython', sys.version.split()[0], 'GIL off')"

rm -rf "$work/venv"
"$py" -m venv --without-pip "$work/venv"
venv_py="$work/venv/bin/python"
# pip into the venv from a wheel (the interpreter is built without ensurepip)
curl -fsSL -o "$work/get-pip.py" https://bootstrap.pypa.io/get-pip.py
"$venv_py" "$work/get-pip.py" --quiet
"$venv_py" -m pip install --quiet --only-binary=:all: pytest numpy "scikit-build-core>=0.10" "nanobind>=3.1,<4"
if [ -n "${RAPTOR_DIR:-}" ]; then
    "$venv_py" -m pip install --quiet --no-deps "$RAPTOR_DIR"
else
    "$venv_py" -m pip install --quiet --only-binary=:all: "raptor-core>=0.3"
fi

# the binding, instrumented, with debug info so reports name file:line
CFLAGS="-fsanitize=thread -g -O2" CXXFLAGS="-fsanitize=thread -g -O2" LDFLAGS="-fsanitize=thread" \
    "$venv_py" -m pip install --quiet --no-deps --no-build-isolation \
    -Ccmake.build-type=RelWithDebInfo -Cinstall.strip=false -Cbuild-dir="build/tsan-{wheel_tag}" \
    -Ccmake.define.EAGLE_PYTHON_CUDA_PLUGIN=OFF -Ccmake.define.EAGLE_PYTHON_MPI=OFF \
    -e "$root/python"
core="$("$venv_py" -c 'import eagle._core as m; print(m.__file__)')"
n_tsan="$(nm -D "$core" | grep -c __tsan_ || true)"
[ "$n_tsan" -gt 0 ] || { echo "RED: $core carries no TSan instrumentation"; exit 1; }
echo "_core instrumented ($n_tsan __tsan_ symbols)"

# (ii) non-vacuity: the canary race must be reported with only CPython's file
rm -rf "$work/reports" && mkdir -p "$work/reports"
TSAN_OPTIONS="suppressions=$cpython_supp exitcode=0 log_path=$work/reports/canary" \
    "$venv_py" - <<'EOF'
import threading
import eagle._core as core
threads = [threading.Thread(target=lambda: [core._unsynchronised_bump() for _ in range(20000)])
           for _ in range(8)]
for t in threads: t.start()
for t in threads: t.join()
EOF
if ! cat "$work"/reports/canary.* 2>/dev/null | grep -q "unsynchronisedBump"; then
    echo "RED: the canary race was not reported — TSan is not reaching _core"; exit 1
fi
echo "canary race reported (detector live)"

# (iii) the ft rows, with every suppression
n_ft="$(cd "$root/python" && "$venv_py" -m pytest tests -m ft --collect-only -q -p no:cacheprovider | grep -c '::' || true)"
[ "$n_ft" -gt 0 ] || { echo "RED: no ft rows collected"; exit 1; }
set +e
(cd "$root/python" && TSAN_OPTIONS="suppressions=$work/all.supp exitcode=0 second_deadlock_stack=1 log_path=$work/reports/ft" \
    "$venv_py" -m pytest tests -m ft -q -rs -p no:cacheprovider --basetemp="$work/pytest")
rc=$?
set -e
n_reports="$(cat "$work"/reports/ft.* 2>/dev/null | grep -c 'WARNING: ThreadSanitizer' || true)"
if [ "$n_reports" -gt 0 ]; then
    echo "RED: $n_reports TSan report(s):"; cat "$work"/reports/ft.*; exit 1
fi
[ "$rc" -eq 0 ] || { echo "RED: ft rows failed (pytest rc=$rc)"; exit 1; }
echo "GREEN: ft rows under TSan ($n_ft collected), 0 reports"
