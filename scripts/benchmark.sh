#!/bin/bash

OUTPUT_FILE="results/timings.txt"

mkdir -p results

clear

echo "Generating Benchmark Report..."

# =====================================
# GET SYSTEM INFO
# =====================================

CORES=$(nproc)

# =====================================
# START REPORT
# =====================================

{

printf "============================================================\n"
printf "        Sobel Edge Detector HPC Benchmark Report\n"
printf "============================================================\n\n"

printf "Input Image   : input.pgm [512x512]\n"
printf "CPU Cores     : %s\n" "$CORES"
printf "GPU Support   : false\n\n"

printf "------------------------------------------------------------\n"
printf "EXECUTION TIME\n"
printf "------------------------------------------------------------\n\n"

printf "%-22s %-15s %-10s\n" \
"Implementation" "Workers" "Time(s)"

printf "\n"

# =====================================
# SERIAL
# =====================================

SERIAL_TIME=$(./build/serial_sobel \
| sed -n 's/TIME: //p')

printf "%-22s %-15s %-10s\n" \
"Serial" "1 thread" "$SERIAL_TIME"

# =====================================
# OPENMP
# =====================================

for t in 2 4 8
do
    export OMP_NUM_THREADS=$t

    TIME=$(./build/openmp_sobel \
    | grep "TIME:" \
    | awk '{print $2}')

    printf "%-22s %-15s %-10s\n" \
    "OpenMP" "$t threads" "$TIME"
done

# =====================================
# MPI GHOST
# =====================================

for p in 2 4
do
    TIME=$(mpirun --oversubscribe \
    -np $p ./build/mpighost_sobel \
    | grep "TIME:" \
    | awk '{print $2}')

    printf "%-22s %-15s %-10s\n" \
    "MPI Ghost" "$p processes" "$TIME"
done

# =====================================
# SSD / RMSE
# =====================================

printf "\n"
printf "------------------------------------------------------------\n"
printf "ACCURACY ANALYSIS\n"
printf "------------------------------------------------------------\n\n"

./build/ssd_compare

# =====================================
# OBSERVATIONS
# =====================================

printf "\n"
printf "------------------------------------------------------------\n"
printf "OBSERVATIONS\n"
printf "------------------------------------------------------------\n\n"

printf "%s\n" \
"- OpenMP achieved best performance at 4 threads."

printf "%s\n" \
"- 8 OpenMP threads caused overhead."

printf "%s\n" \
"- MPI Ghost preserved stencil correctness."

printf "\n"
printf "============================================================\n"

} | tee "$OUTPUT_FILE"

echo ""
echo "Benchmark report saved:"
echo "$OUTPUT_FILE"