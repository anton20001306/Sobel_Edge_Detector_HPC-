#!/bin/bash
# =============================================================================
# run_multi_res.sh
# Runs the existing run_pipeline.sh on 3 resized versions of the same image,
# then calls make_plots.py to generate comparison plots for the report.
#
# EE7218 – High Performance Computing | Group 02
#
# USAGE:
#   chmod +x run_multi_res.sh
#   ./run_multi_res.sh <input_image>
#
# EXAMPLE:
#   ./run_multi_res.sh animal.jpeg
#
# WHAT IT DOES:
#   1. Creates 3 resized versions of your input  (25%, 50%, 100%)
#   2. Runs ./run_pipeline.sh on each            (one full benchmark per size)
#   3. Saves each results_report.txt under multi_res_results/
#   4. Calls make_plots.py to produce comparison plots in plots/
#
# REQUIREMENTS:
#   - run_pipeline.sh, make_plots.py, and all *.c sources in the current dir
#   - imagemagick (convert, identify), python3 with matplotlib + numpy
#
# RUNTIME:
#   Each pipeline run takes a few minutes (depends on image size and
#   number of test configurations). Total ≈ 10–20 minutes for 3 sizes.
# =============================================================================

set -e

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'

# ── Arg check ────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo -e "${RED}ERROR: No input image specified.${NC}"
    echo "Usage:   $0 <input_image>"
    echo "Example: $0 animal.jpeg"
    exit 1
fi

INPUT="$1"
[[ ! -f "$INPUT" ]] && { echo -e "${RED}File not found: $INPUT${NC}"; exit 1; }
[[ ! -f "./run_pipeline.sh" ]] && { echo -e "${RED}run_pipeline.sh not found in current directory.${NC}"; exit 1; }
[[ ! -f "./make_plots.py"   ]] && { echo -e "${RED}make_plots.py not found in current directory.${NC}"; exit 1; }
[[ ! -x "./run_pipeline.sh" ]] && chmod +x ./run_pipeline.sh

# ── Configuration ────────────────────────────────────────────────────────────
SIZES=("small:25%" "medium:50%" "large:100%")   # label:scale
RESULTS_DIR="multi_res_results"
INPUTS_DIR="multi_res_inputs"
EXT="${INPUT##*.}"

mkdir -p "$INPUTS_DIR" "$RESULTS_DIR"

# ── Banner ───────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Multi-Resolution Benchmark – Parallel Sobel Edge Detector  ║"
echo "║   EE7218 High Performance Computing | Group 02               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── STEP 1: Create resized versions ──────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}▶  STEP 1 — Creating resized variants of: ${INPUT}${NC}"
echo "──────────────────────────────────────────────────────────────"
for entry in "${SIZES[@]}"; do
    label="${entry%%:*}"
    scale="${entry##*:}"
    out="$INPUTS_DIR/${label}.png"
    convert "$INPUT" -resize "$scale" "$out"
    dims=$(identify -format "%wx%h" "$out")
    echo -e "  ${GREEN}✔${NC}  $label  →  $out  [$dims]"
done

# ── STEP 2: Run pipeline for each size ───────────────────────────────────────
for entry in "${SIZES[@]}"; do
    label="${entry%%:*}"
    f="$INPUTS_DIR/${label}.png"

    echo -e "\n${BOLD}${CYAN}▶  STEP 2.${label} — Running pipeline on ${label} image${NC}"
    echo "════════════════════════════════════════════════════════════════"

    # Run the existing pipeline (it produces results_report.txt + pipeline_output/)
    ./run_pipeline.sh "$f"

    # Archive results for this size
    cp results_report.txt "$RESULTS_DIR/results_${label}.txt"
    [[ -d pipeline_output ]] && {
        rm -rf "$RESULTS_DIR/pipeline_${label}"
        cp -r pipeline_output "$RESULTS_DIR/pipeline_${label}"
    }
    echo -e "\n  ${GREEN}✔${NC}  Saved → $RESULTS_DIR/results_${label}.txt"
done

# ── STEP 3: Generate comparison plots ────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}▶  STEP 3 — Generating comparison plots${NC}"
echo "──────────────────────────────────────────────────────────────"
python3 make_plots.py \
    "$RESULTS_DIR/results_small.txt" \
    "$RESULTS_DIR/results_medium.txt" \
    "$RESULTS_DIR/results_large.txt"

# ── Done ─────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  Multi-resolution run complete!  ✔           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  ${BOLD}Results files  :${NC} $RESULTS_DIR/results_{small,medium,large}.txt"
echo -e "  ${BOLD}Plots (PNG/PDF):${NC} plots/"
echo -e "  ${BOLD}For your report:${NC} insert any of the .png or .pdf files from plots/"
echo ""
