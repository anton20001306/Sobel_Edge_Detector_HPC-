#!/bin/bash
# =============================================================================
# run_pipeline.sh
# Full Benchmark Pipeline – Parallel Sobel Edge Detector
#
# EE7218 – High Performance Computing | Group 02
# Members: EG/2021/4512 W.A.P.N Fernando | EG/2021/4654 S.W.M Madhusan
#
# USAGE:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh <input_image>    (any format: jpg, png, bmp, tiff, pgm...)
#
# EXAMPLES:
#   ./run_pipeline.sh photo.jpg
#   ./run_pipeline.sh picture.png
#   ./run_pipeline.sh scan.tiff
#
# WHAT IT DOES:
#   1.  Checks all required tools are installed
#   2.  Converts your image to grayscale PGM (P5) automatically
#   3.  Compiles all 4 implementations (serial, OpenMP, MPI, CUDA hybrid)
#   4.  Runs serial baseline (x1)
#   5.  Runs OpenMP with 1, 2, 4, 8, 16 threads
#   6.  Runs MPI   with 1, 2, 4, 8 processes
#   7.  Runs MPI+CUDA hybrid with 4 ranks (skipped if no GPU)
#   8.  Computes RMSE accuracy vs serial baseline for all outputs
#   9.  Prints full formatted results table (copy into your report)
#  10.  Saves everything to results_report.txt
#  11.  Converts all PGM outputs to PNG for easy viewing
#
# REQUIREMENTS (install with apt-get):
#   gcc, mpicc, mpirun, imagemagick (convert), python3, python3-numpy
#   nvcc + CUDA Toolkit (optional – hybrid skipped automatically if absent)
#
# NOTE ON mpirun FLAGS USED:
#   --allow-run-as-root  : allows MPI to run as root (required on many clusters)
#   --oversubscribe      : allows more ranks than physical cores
#   --mca plm_rsh_agent "": disables SSH launcher (use shared-memory transport)
#   Adjust these for your cluster's scheduler (SLURM, PBS, etc.)
# =============================================================================

set -e

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m';  GREEN='\033[0;32m';  YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m';   BOLD='\033[1m';  NC='\033[0m'

# ── Configuration (edit these to match your system) ───────────────────────────
OMP_THREADS=(1 2 4 8 16)         # OpenMP thread counts to test
MPI_PROCS=(1 2 4 8)              # MPI process counts to test
REPEATS=3                        # Runs per test; median is reported
RUN_TIMEOUT=120                  # Seconds before a single run is killed
OUTPUT_DIR="pipeline_output"     # All output files go here
RESULTS_FILE="results_report.txt"

# mpirun flags – adjust for your cluster
MPIRUN_FLAGS="--oversubscribe --mca plm_rsh_agent \"\""
# If running as root (common on HPC login nodes):
# MPIRUN_FLAGS="--allow-run-as-root --oversubscribe --mca plm_rsh_agent \"\""

# ── Helpers ───────────────────────────────────────────────────────────────────
print_banner() {
    echo -e "${BOLD}${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     Parallel Sobel Edge Detector – Full Benchmark Pipeline   ║"
    echo "║     EE7218 High Performance Computing | Group 02             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() { echo -e "\n${BOLD}${CYAN}▶  $1${NC}\n────────────────────────────────────────────────────────────────"; }
print_ok()   { echo -e "  ${GREEN}✔${NC}  $1"; }
print_warn() { echo -e "  ${YELLOW}⚠${NC}  $1"; }
print_err()  { echo -e "  ${RED}✘${NC}  $1"; }
print_info() { echo -e "  ${BLUE}ℹ${NC}  $1"; }

# Extract timing from program output (looks for "X.XXXXXX seconds")
extract_time() { echo "$1" | grep -oP '[\d]+\.[\d]+(?= seconds)' | head -1; }

# Run a program REPEATS times and set variable to the median time
# Usage: run_median <varname> <program_output_line_prefix> <command...>
run_median() {
    local _var=$1; local _label=$2; shift 2
    local _times=() _t _raw
    for (( r=0; r<REPEATS; r++ )); do
        _raw=$(timeout $RUN_TIMEOUT "$@" 2>&1) || { echo -e "  ${YELLOW}⚠${NC}  $_label run $((r+1)) timed out or failed"; continue; }
        _t=$(extract_time "$_raw")
        [[ -n "$_t" ]] && _times+=("$_t")
    done
    if [[ ${#_times[@]} -eq 0 ]]; then
        eval "$_var='FAILED'"
    else
        local _sorted _mid
        _sorted=$(printf '%s\n' "${_times[@]}" | sort -n)
        _mid=$(echo "$_sorted" | sed -n "$(( (REPEATS-1)/2 + 1 ))p")
        eval "$_var='$_mid'"
    fi
}

# Run mpirun with timeout (mpirun ignores plain timeout sometimes)
run_mpi_median() {
    local _var=$1; local _label=$2; local _np=$3; shift 3
    local _times=() _t _raw
    for (( r=0; r<REPEATS; r++ )); do
        _raw=$(timeout $RUN_TIMEOUT mpirun $MPIRUN_FLAGS -np "$_np" "$@" 2>&1) || {
            echo -e "  ${YELLOW}⚠${NC}  $_label run $((r+1)) timed out or failed (check mpirun flags)"
            continue
        }
        _t=$(extract_time "$_raw")
        [[ -n "$_t" ]] && _times+=("$_t")
    done
    if [[ ${#_times[@]} -eq 0 ]]; then
        eval "$_var='FAILED'"
    else
        local _sorted _mid
        _sorted=$(printf '%s\n' "${_times[@]}" | sort -n)
        _mid=$(echo "$_sorted" | sed -n "$(( (REPEATS-1)/2 + 1 ))p")
        eval "$_var='$_mid'"
    fi
}

# Compute speedup: T_serial / T_parallel  (returns N/A if either is missing)
calc_speedup() {
    local _serial=$1 _parallel=$2
    if [[ "$_serial" == "FAILED" || "$_parallel" == "FAILED" || -z "$_serial" || -z "$_parallel" ]]; then
        echo "N/A"
    else
        python3 -c "print(f'{float(\"$_serial\")/float(\"$_parallel\"):.4f}')" 2>/dev/null || echo "N/A"
    fi
}

# Compute efficiency: speedup / workers * 100%
calc_efficiency() {
    local _speedup=$1 _workers=$2
    if [[ "$_speedup" == "N/A" ]] || ! [[ "$_workers" =~ ^[0-9]+$ ]]; then
        echo "N/A"
    else
        python3 -c "print(f'{float(\"$_speedup\")/$_workers*100:.1f}%')" 2>/dev/null || echo "N/A"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
print_banner

# ── Check argument ─────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo -e "${RED}ERROR: No input image specified.${NC}"
    echo "Usage:   $0 <input_image>"
    echo "Example: $0 photo.jpg"
    exit 1
fi
INPUT_IMAGE="$1"
[[ ! -f "$INPUT_IMAGE" ]] && { print_err "File not found: $INPUT_IMAGE"; exit 1; }

# ── STEP 0: Check dependencies ────────────────────────────────────────────────
print_step "STEP 0 – Checking Dependencies"

check_dep() {
    if command -v "$1" &>/dev/null; then
        print_ok "$1  →  $(command -v $1)"
        return 0
    else
        [[ "$2" == "required" ]] && { print_err "$1 NOT FOUND — install: $3"; exit 1; }
        print_warn "$1 NOT FOUND (optional — $2 will be skipped)"
        return 1
    fi
}

check_dep gcc      required  "sudo apt-get install gcc"
check_dep mpicc    required  "sudo apt-get install libopenmpi-dev openmpi-bin"
check_dep mpirun   required  "sudo apt-get install openmpi-bin"
check_dep convert  required  "sudo apt-get install imagemagick"
check_dep python3  required  "sudo apt-get install python3"

HAVE_CUDA=false; HAVE_HYBRID=false
if check_dep nvcc optional "CUDA hybrid"; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        HAVE_CUDA=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        print_ok "CUDA GPU: $GPU_NAME"
    else
        print_warn "nvcc found but no CUDA GPU detected — hybrid skipped"
    fi
fi

python3 -c "import numpy" 2>/dev/null && print_ok "python3-numpy  →  OK" || {
    print_warn "numpy not found — installing..."
    if command -v pip3 &>/dev/null; then
        pip3 install numpy --break-system-packages -q
    elif command -v pip &>/dev/null; then
        pip install numpy --break-system-packages -q
    else
        python3 -m pip install numpy --break-system-packages -q 2>/dev/null || \
        sudo apt-get install -y python3-numpy -q 2>/dev/null || {
            print_err "Could not install numpy. Run: sudo apt-get install python3-numpy"
            exit 1
        }
    fi
}

MAX_CORES=$(nproc 2>/dev/null || echo 4)
print_info "CPU cores: $MAX_CORES  |  GPU: $HAVE_CUDA  |  Timeout per run: ${RUN_TIMEOUT}s  |  Repeats: $REPEATS"

# ── STEP 1: Setup output directory ────────────────────────────────────────────
print_step "STEP 1 – Preparing Output Directory"
mkdir -p "$OUTPUT_DIR"
print_ok "Output directory: $OUTPUT_DIR/"

# ── STEP 2: Convert image to grayscale PGM ────────────────────────────────────
print_step "STEP 2 – Converting Input Image → Grayscale PGM"

INPUT_PGM="$OUTPUT_DIR/input.pgm"
print_info "Input  : $INPUT_IMAGE"
print_info "Output : $INPUT_PGM"

# Convert to binary (P5) grayscale PGM using ImageMagick
convert "$INPUT_IMAGE" -colorspace Gray -type Grayscale -compress None "$INPUT_PGM" 2>/dev/null || {
    print_err "ImageMagick conversion failed"
    exit 1
}

# Ensure output is P5 (binary), not P2 (ASCII)
PGM_HDR=$(head -c 2 "$INPUT_PGM" 2>/dev/null || echo "XX")
if [[ "$PGM_HDR" == "P2" ]]; then
    print_warn "Got P2 (ASCII PGM) — converting to P5 (binary)..."
    python3 - "$INPUT_PGM" << 'PYCONV'
import sys, struct
path = sys.argv[1]
with open(path, 'r') as f:
    tokens = f.read().split()
assert tokens[0] == 'P2'
w, h, maxv = int(tokens[1]), int(tokens[2]), int(tokens[3])
pixels = bytes([int(x) for x in tokens[4:]])
with open(path, 'wb') as out:
    out.write(f'P5\n{w} {h}\n{maxv}\n'.encode())
    out.write(pixels)
print("P2→P5 conversion done")
PYCONV
fi

IMG_DIMS=$(python3 -c "
with open('$INPUT_PGM','rb') as f:
    f.readline()
    line = f.readline()
    while line.startswith(b'#'): line = f.readline()
    w,h = map(int, line.split())
    print(f'{w}x{h}')
" 2>/dev/null || echo "unknown")

print_ok "Grayscale PGM ready: $INPUT_PGM  [$IMG_DIMS]"

# ── STEP 3: Compile all implementations ───────────────────────────────────────
print_step "STEP 3 – Compiling"

COMPILE_LOG="$OUTPUT_DIR/compile.log"
> "$COMPILE_LOG"
ALL_OK=true

compile_one() {
    local name=$1; local out=$2; shift 2
    echo -n "  Compiling $name ... "
    if "$@" -o "$out" >> "$COMPILE_LOG" 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}  (see $COMPILE_LOG)"
        ALL_OK=false
    fi
}

compile_one "serial_sobel"  "$OUTPUT_DIR/serial_sobel"  gcc -O2 -fopenmp serial_sobel.c -lm
compile_one "omp_sobel"     "$OUTPUT_DIR/omp_sobel"     gcc -O2 -fopenmp omp_sobel.c -lm
compile_one "mpi_sobel"     "$OUTPUT_DIR/mpi_sobel"     mpicc -O2 mpi_sobel.c -lm

if [[ "$HAVE_CUDA" == "true" ]]; then
    MPI_CF=$(mpicc --showme:compile 2>/dev/null || echo "")
    MPI_LF=$(mpicc --showme:link   2>/dev/null || echo "-lmpi")
    echo -n "  Compiling hybrid_sobel ... "
    if nvcc -O2 hybrid_sobel.cu $MPI_CF $MPI_LF -lm -o "$OUTPUT_DIR/hybrid_sobel" >> "$COMPILE_LOG" 2>&1; then
        echo -e "${GREEN}OK${NC}"; HAVE_HYBRID=true
    else
        echo -e "${RED}FAILED${NC}  (see $COMPILE_LOG)"
    fi
else
    print_warn "Skipping hybrid_sobel (no CUDA)"
fi

[[ "$ALL_OK" == "false" ]] && { print_err "Compilation failed — check $COMPILE_LOG"; exit 1; }
print_ok "All required binaries compiled."

# ── STEP 4: Run benchmarks ────────────────────────────────────────────────────
print_step "STEP 4 – Running Benchmarks  (each test ×$REPEATS, reporting median)"

declare -A TIME SP EFF   # timing, speedup, efficiency
cd "$OUTPUT_DIR"

# ── 4a Serial ─────────────────────────────────────────────────────────────────
echo -e "\n  ${BOLD}[Serial]${NC}"
run_median TIME_SERIAL "Serial" ./serial_sobel input.pgm output_serial.pgm
print_ok "Serial  [1 CPU]  →  ${TIME_SERIAL}s"
TIME["serial_1"]=$TIME_SERIAL
SP["serial_1"]="1.0000"
EFF["serial_1"]="100.0%"

# ── 4b OpenMP ─────────────────────────────────────────────────────────────────
echo -e "\n  ${BOLD}[OpenMP]${NC}"
for T in "${OMP_THREADS[@]}"; do
    OUT="output_omp_${T}t.pgm"
    run_median T_NOW "OpenMP-$T" ./omp_sobel input.pgm "$OUT" "$T"
    SP_NOW=$(calc_speedup "$TIME_SERIAL" "$T_NOW")
    EFF_NOW=$(calc_efficiency "$SP_NOW" "$T")
    TIME["omp_$T"]=$T_NOW
    SP["omp_$T"]=$SP_NOW
    EFF["omp_$T"]=$EFF_NOW
    printf "  ${GREEN}✔${NC}  OpenMP  [%2d threads]  →  %10s s   speedup: %6sx   efficiency: %s\n" \
        "$T" "$T_NOW" "$SP_NOW" "$EFF_NOW"
done

# ── 4c MPI ────────────────────────────────────────────────────────────────────
echo -e "\n  ${BOLD}[MPI]${NC}"
for P in "${MPI_PROCS[@]}"; do
    OUT="output_mpi_${P}p.pgm"
    run_mpi_median T_NOW "MPI-$P" "$P" ./mpi_sobel input.pgm "$OUT"
    SP_NOW=$(calc_speedup "$TIME_SERIAL" "$T_NOW")
    EFF_NOW=$(calc_efficiency "$SP_NOW" "$P")
    TIME["mpi_$P"]=$T_NOW
    SP["mpi_$P"]=$SP_NOW
    EFF["mpi_$P"]=$EFF_NOW
    printf "  ${GREEN}✔${NC}  MPI     [%2d processes]  →  %10s s   speedup: %6sx   efficiency: %s\n" \
        "$P" "$T_NOW" "$SP_NOW" "$EFF_NOW"
done

# ── 4d Hybrid MPI+CUDA ────────────────────────────────────────────────────────
T_HYBRID="N/A"; SP_HYBRID="N/A"; EFF_HYBRID="N/A"
H2D_TIME="N/A"; D2H_TIME="N/A"

if [[ "$HAVE_HYBRID" == "true" ]]; then
    echo -e "\n  ${BOLD}[Hybrid MPI+CUDA]${NC}"
    HYBRID_TIMES=()
    for (( r=0; r<REPEATS; r++ )); do
        RAW=$(timeout $RUN_TIMEOUT mpirun $MPIRUN_FLAGS -np 4 \
              ./hybrid_sobel input.pgm output_hybrid.pgm 2>&1) || continue
        t=$(extract_time "$RAW")
        [[ -n "$t" ]] && HYBRID_TIMES+=("$t")
        H2D_TIME=$(echo "$RAW" | grep -oP 'H2D.*?:\s+\K[\d.]+' | head -1 || echo "N/A")
        D2H_TIME=$(echo "$RAW" | grep -oP 'D2H.*?:\s+\K[\d.]+' | head -1 || echo "N/A")
    done
    if [[ ${#HYBRID_TIMES[@]} -gt 0 ]]; then
        T_HYBRID=$(printf '%s\n' "${HYBRID_TIMES[@]}" | sort -n | \
                   sed -n "$(( (REPEATS-1)/2 + 1 ))p")
        SP_HYBRID=$(calc_speedup "$TIME_SERIAL" "$T_HYBRID")
        EFF_HYBRID=$(calc_efficiency "$SP_HYBRID" "4")
    fi
    print_ok "Hybrid  [4 ranks×GPU]  →  ${T_HYBRID}s   speedup: ${SP_HYBRID}x   efficiency: $EFF_HYBRID"
    print_info "GPU transfer: H2D = ${H2D_TIME}ms  |  D2H = ${D2H_TIME}ms"
fi

cd ..

# ── STEP 5: Accuracy – RMSE vs serial ─────────────────────────────────────────
print_step "STEP 5 – Accuracy Check (RMSE vs Serial Baseline)"

RMSE_OUTPUT=$(python3 << PYEOF
import os, glob
import numpy as np

def read_pgm(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'P5', f"Not P5: {path}"
        line = f.readline()
        while line.startswith(b'#'): line = f.readline()
        w, h = map(int, line.split())
        f.readline()  # maxval
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w).astype(np.float64)

base = '$OUTPUT_DIR/output_serial.pgm'
if not os.path.exists(base):
    print("  Serial output not found")
    exit(0)

serial = read_pgm(base)
outputs = sorted(glob.glob('$OUTPUT_DIR/output_*.pgm'))
print(f"  {'Output File':<38} {'RMSE':>8}  {'Max Diff':>10}  {'Status'}")
print(f"  {'-'*38} {'-'*8}  {'-'*10}  {'-'*8}")

for path in outputs:
    fname = os.path.basename(path)
    if fname == 'output_serial.pgm': continue
    try:
        img = read_pgm(path)
        if img.shape != serial.shape:
            print(f"  {fname:<38} shape mismatch – skipped")
            continue
        rmse = float(np.sqrt(np.mean((serial - img)**2)))
        maxd = float(np.max(np.abs(serial - img)))
        status = 'PASS ✔' if rmse < 1.0 else 'CHECK !'
        print(f"  {fname:<38} {rmse:>8.4f}  {maxd:>10.1f}  {status}")
    except Exception as e:
        print(f"  {fname:<38} ERROR: {e}")
PYEOF
)
echo "$RMSE_OUTPUT"

# ── STEP 6: Print results table ───────────────────────────────────────────────
print_step "STEP 6 – Full Results Table  (copy into your report)"

echo ""
echo -e "  ${BOLD}Image:${NC} $INPUT_IMAGE  →  [$IMG_DIMS] grayscale PGM"
echo -e "  ${BOLD}Serial baseline:${NC} ${TIME_SERIAL}s   |   Repeats: $REPEATS (median)   |   Cores: $MAX_CORES"
echo ""
printf "  ${BOLD}%-26s %-11s %-14s %-12s %-12s${NC}\n" \
    "Implementation" "Workers" "Time (s)" "Speedup" "Efficiency"
printf "  %-26s %-11s %-14s %-12s %-12s\n" \
    "──────────────────────────" "───────────" "──────────────" "────────────" "────────────"
printf "  %-26s %-11s %-14s %-12s %-12s\n" \
    "Serial (baseline)" "1 thread" "${TIME_SERIAL}" "1.0000 x" "100.0%"
echo ""
for T in "${OMP_THREADS[@]}"; do
    printf "  %-26s %-11s %-14s %-12s %-12s\n" \
        "OpenMP" "$T threads" "${TIME["omp_$T"]:-N/A}" "${SP["omp_$T"]:-N/A} x" "${EFF["omp_$T"]:-N/A}"
done
echo ""
for P in "${MPI_PROCS[@]}"; do
    printf "  %-26s %-11s %-14s %-12s %-12s\n" \
        "MPI" "$P processes" "${TIME["mpi_$P"]:-N/A}" "${SP["mpi_$P"]:-N/A} x" "${EFF["mpi_$P"]:-N/A}"
done
if [[ "$HAVE_HYBRID" == "true" ]]; then
    echo ""
    printf "  %-26s %-11s %-14s %-12s %-12s\n" \
        "MPI+CUDA Hybrid" "4 ranks" "$T_HYBRID" "$SP_HYBRID x" "$EFF_HYBRID"
    echo ""
    echo -e "  ${BLUE}ℹ  GPU Transfer Overhead:  H2D = ${H2D_TIME} ms  |  D2H = ${D2H_TIME} ms${NC}"
fi
echo ""

# ── STEP 7: Save results file ─────────────────────────────────────────────────
print_step "STEP 7 – Saving Results"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

{
cat << HDREOF
================================================================================
  PARALLEL SOBEL EDGE DETECTOR – BENCHMARK RESULTS
  EE7218 High Performance Computing | Group 02
  EG/2021/4512 W.A.P.N Fernando  |  EG/2021/4654 S.W.M Madhusan
  Generated : $TIMESTAMP
================================================================================

  Input image     : $INPUT_IMAGE
  Processed as    : $INPUT_PGM  [$IMG_DIMS]
  Serial baseline : ${TIME_SERIAL} seconds
  Repeats / run   : $REPEATS  (median reported)
  CPU cores       : $MAX_CORES
  GPU support     : $HAVE_CUDA

────────────────────────────────────────────────────────────────────────────────
  EXECUTION TIME  (computation only — I/O excluded)
────────────────────────────────────────────────────────────────────────────────
$(printf '%-26s %-11s %-14s %-12s %-12s\n' 'Implementation' 'Workers' 'Time (s)' 'Speedup' 'Efficiency')
$(printf '%-26s %-11s %-14s %-12s %-12s\n' '──────────────────────────' '───────────' '──────────────' '────────────' '────────────')
$(printf '%-26s %-11s %-14s %-12s %-12s\n' 'Serial (baseline)' '1 thread' "${TIME_SERIAL}" '1.0000 x' '100.0%')
HDREOF

for T in "${OMP_THREADS[@]}"; do
    printf '%-26s %-11s %-14s %-12s %-12s\n' \
        "OpenMP" "$T threads" "${TIME["omp_$T"]:-N/A}" "${SP["omp_$T"]:-N/A} x" "${EFF["omp_$T"]:-N/A}"
done
echo ""
for P in "${MPI_PROCS[@]}"; do
    printf '%-26s %-11s %-14s %-12s %-12s\n' \
        "MPI" "$P processes" "${TIME["mpi_$P"]:-N/A}" "${SP["mpi_$P"]:-N/A} x" "${EFF["mpi_$P"]:-N/A}"
done
if [[ "$HAVE_HYBRID" == "true" ]]; then
    echo ""
    printf '%-26s %-11s %-14s %-12s %-12s\n' "MPI+CUDA Hybrid" "4 ranks" "$T_HYBRID" "$SP_HYBRID x" "$EFF_HYBRID"
    echo ""
    echo "  GPU Transfer Overhead:  H2D = ${H2D_TIME} ms  |  D2H = ${D2H_TIME} ms"
fi

cat << ACCEOF

────────────────────────────────────────────────────────────────────────────────
  ACCURACY  (RMSE vs serial baseline — lower is better, 0 = identical)
────────────────────────────────────────────────────────────────────────────────
$RMSE_OUTPUT

────────────────────────────────────────────────────────────────────────────────
  OUTPUT FILES
────────────────────────────────────────────────────────────────────────────────
ACCEOF
ls -lh "$OUTPUT_DIR"/*.pgm 2>/dev/null || echo "  (none)"
echo ""
echo "  Pipeline completed: $TIMESTAMP"
echo "================================================================================"
} > "$RESULTS_FILE"

print_ok "Results saved → $RESULTS_FILE"



# ── STEP 8: Convert PGM outputs to PNG ────────────────────────────────────────
print_step "STEP 8 – Converting Outputs to PNG (for viewing)"

PNG_DIR="$OUTPUT_DIR/png_previews"
mkdir -p "$PNG_DIR"
COUNT=0
for pgm in "$OUTPUT_DIR"/output_*.pgm; do
    base=$(basename "$pgm" .pgm)
    if convert "$pgm" "$PNG_DIR/$base.png" 2>/dev/null; then
        print_ok "$PNG_DIR/$base.png"
        COUNT=$((COUNT+1))
    else
        print_warn "Could not convert $pgm"
    fi
done
print_info "$COUNT PNG previews saved to $PNG_DIR/"



# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   Pipeline Complete!  ✔                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  ${BOLD}Results table  :${NC} $RESULTS_FILE"
echo -e "  ${BOLD}PGM outputs    :${NC} $OUTPUT_DIR/"
echo -e "  ${BOLD}PNG previews   :${NC} $PNG_DIR/"
echo ""