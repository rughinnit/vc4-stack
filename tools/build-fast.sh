#!/bin/bash
# build-fast.sh -- memory-aware parallel build with ccache for the VC4 stack.
#
# On a 1 GB Pi 3B the safe build parallelism is set by RAM, not by core count.
# Measured on this board (see docs/BUILD.md):
#   * engine (plain C, -O2):           ~45 MiB per job, 171 MiB for -j4 total
#   * VC4C   (C++ + LLVM headers):    ~400 MiB per job, 673 MiB for -j4 plus
#                                     800 MiB of swap -> unsafe
# This script picks the largest job count that still fits in MemAvailable and
# uses ccache when it is installed.
#
# Usage:
#   build-fast.sh [DIR] [--clean] [-jN] [-- <extra build args>]
#
#   DIR               project directory (default: current directory)
#   --clean           clean before building
#   -jN               force N jobs (overrides the memory calculation)
#   -- <args...>      passed to make / cmake --build
#
# Environment:
#   BUILD_FAST_JOBS       force job count (same as -jN)
#   BUILD_FAST_RESERVE    MiB to keep free (default 200)
#   BUILD_FAST_PER_JOB    MiB assumed per compiler job (default: auto)
#   BUILD_FAST_NO_CCACHE  set to 1 to ignore ccache
#   BUILD_FAST_TARGET     build target passed to make (default: all)
#
# Exit status is the build's exit status.
set -u

DIR="."
CLEAN=0
JOBS="${BUILD_FAST_JOBS:-}"
EXTRA=()
TARGET="${BUILD_FAST_TARGET:-all}"

while [ $# -gt 0 ]; do
    case "$1" in
        --clean) CLEAN=1 ;;
        -j[0-9]*) JOBS="${1#-j}" ;;
        -j) shift; JOBS="$1" ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        -*) echo "build-fast: unknown option $1" >&2; exit 2 ;;
        *) DIR="$1" ;;
    esac
    shift
done

cd "$DIR" || { echo "build-fast: no such directory: $DIR" >&2; exit 2; }
DIR=$(pwd)
NPROC=$(nproc)
RESERVE="${BUILD_FAST_RESERVE:-200}"

mem_avail() { awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo; }
AVAIL=$(mem_avail)
SWAP0=$(free -m | awk '/^Swap:/{print $3}')

# --- build system -------------------------------------------------------
if [ -f CMakeCache.txt ]; then
    MODE=cmake; BUILD_DIR="$DIR"
elif [ -f CMakeLists.txt ] && [ -d build ] && [ -f build/CMakeCache.txt ]; then
    MODE=cmake; BUILD_DIR="$DIR/build"
elif [ -f Makefile ] || [ -f makefile ]; then
    MODE=make; BUILD_DIR="$DIR"
else
    echo "build-fast: no Makefile/CMakeLists.txt in $DIR" >&2
    exit 2
fi

# --- per-job memory class ----------------------------------------------
# A project that compiles C++ is treated as LLVM-weight (VC4C); a pure C
# project like the engine is light.  For a CMake build directory the sources
# live elsewhere, so use the cache to find the source tree and the language.
if [ "$MODE" = cmake ]; then
    SRCROOT=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null)
    [ -z "$SRCROOT" ] && SRCROOT=$DIR
    if grep -q '^CMAKE_CXX_COMPILER:' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
        HEAVY=1
    else
        HEAVY=0
    fi
else
    SRCROOT=$DIR
    HEAVY=0
    find "$DIR" -maxdepth 3 \( -name build -o -name .git \) -prune -o \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print -quit 2>/dev/null | grep -q . && HEAVY=1
fi

if [ -z "${BUILD_FAST_PER_JOB:-}" ]; then
    if [ "$HEAVY" = 1 ]; then
        PER_JOB=350
    else
        PER_JOB=60
    fi
else
    PER_JOB="$BUILD_FAST_PER_JOB"
fi

# --- job count ----------------------------------------------------------
if [ -z "$JOBS" ]; then
    FIT=$(( (AVAIL - RESERVE) / PER_JOB ))
    [ "$FIT" -lt 1 ] && FIT=1
    JOBS=$(( FIT < NPROC ? FIT : NPROC ))
fi
case "$JOBS" in ''|*[!0-9]*) echo "build-fast: invalid job count '$JOBS'" >&2; exit 2 ;; esac
[ "$JOBS" -lt 1 ] && JOBS=1

# --- ccache -------------------------------------------------------------
CCACHE=""
if [ "${BUILD_FAST_NO_CCACHE:-0}" != 1 ] && command -v ccache >/dev/null 2>&1; then
    CCACHE=$(command -v ccache)
fi

HITS0=""
[ -n "$CCACHE" ] && HITS0=$(ccache -s 2>/dev/null | awk -F: '/^[[:space:]]*Hits:/{gsub(/ /,"",$2); split($2,a,"/"); print a[1]}')

echo "build-fast: $DIR"
echo "build-fast: mode=$MODE jobs=$JOBS/$NPROC mem_avail=${AVAIL}MiB reserve=${RESERVE}MiB mem_per_job=${PER_JOB}MiB"
if [ -n "$CCACHE" ]; then
    if [ "$MODE" = cmake ]; then
        echo "build-fast: ccache=$CCACHE (CMAKE_*_COMPILER_LAUNCHER)"
    else
        echo "build-fast: ccache=$CCACHE (CC/CXX launcher)"
    fi
else
    echo "build-fast: ccache=no"
fi
[ "$JOBS" -lt "$NPROC" ] && \
    echo "build-fast: note: memory-limited to $JOBS job(s); $NPROC cores, ${AVAIL}MiB available"

# --- configure ccache for CMake ----------------------------------------
if [ "$MODE" = cmake ] && [ -n "$CCACHE" ] && [ "$CLEAN" = 0 ]; then
    if ! grep -q "CMAKE_CXX_COMPILER_LAUNCHER:.*ccache" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
        echo "build-fast: adding ccache launchers to the existing CMake cache"
        ( cd "$BUILD_DIR" && cmake \
              -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
              -DCMAKE_C_COMPILER_LAUNCHER=ccache . >/dev/null ) || {
            echo "build-fast: warning: cmake reconfigure failed, continuing without ccache" >&2
        }
    fi
fi

T0=$(date +%s%N)

if [ "$MODE" = cmake ]; then
    [ "$CLEAN" = 1 ] && cmake --build "$BUILD_DIR" --target clean >/dev/null 2>&1
    if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        cmake -S "$DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
              ${CCACHE:+-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache} \
              || exit $?
    fi
    TGT=()
    [ "$TARGET" != all ] && TGT=(--target "$TARGET")
    ( cd "$BUILD_DIR" && cmake --build . "${TGT[@]}" -- -j"$JOBS" "${EXTRA[@]}" )
    RC=$?
else
    [ "$CLEAN" = 1 ] && make clean >/dev/null 2>&1
    if [ -n "$CCACHE" ]; then
        make -j"$JOBS" CC="ccache gcc" CXX="ccache g++" "$TARGET" "${EXTRA[@]}"
    else
        make -j"$JOBS" "$TARGET" "${EXTRA[@]}"
    fi
    RC=$?
fi

T1=$(date +%s%N)
ELAPSED=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.1f", (b-a)/1e9}')
AVAIL2=$(mem_avail)
SWAP1=$(free -m | awk '/^Swap:/{print $3}')

echo "build-fast: rc=$RC elapsed=${ELAPSED}s mem_avail=${AVAIL2}MiB swap_used=$((SWAP1-SWAP0))MiB_delta"
if [ -n "$CCACHE" ]; then
    ccache -s 2>/dev/null | awk -F: '
        /^[[:space:]]*Hits:/{gsub(/ /,"",$2); h=$2}
        /^[[:space:]]*Misses:/{gsub(/ /,"",$2); m=$2}
        END{if(h!=""||m!="") printf "build-fast: ccache hits=%s misses=%s\n", h, m}'
fi
exit "$RC"
