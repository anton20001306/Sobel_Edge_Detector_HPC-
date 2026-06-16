#!/bin/bash

# =======================================
# OPEN IN NEW TERMINAL WINDOW
# =======================================

if [ -z "$PIPELINE_TERMINAL" ]; then

    export PIPELINE_TERMINAL=1

    gnome-terminal -- \
    bash -c "./scripts/pipeline.sh; exec bash"

    exit
fi

# =======================================
# PIPELINE START
# =======================================

echo "======================================="
echo " Sobel Edge Detector HPC Pipeline"
echo "======================================="

# =======================================
# CREATE REQUIRED FOLDERS
# =======================================

mkdir -p images
mkdir -p build
mkdir -p results

# =======================================
# SELECT IMAGE
# =======================================

echo ""
echo "Select an image file..."

IMAGE=$(zenity --file-selection \
--title="Choose an Image")

# Check cancel
if [ -z "$IMAGE" ]; then
    echo "No image selected."
    exit 1
fi

echo ""
echo "Selected Image:"
echo "$IMAGE"

# =======================================
# CONVERT IMAGE TO PGM
# =======================================

echo ""
echo "Converting image to PGM..."

convert "$IMAGE" \
-grayscale Rec709Luminance \
-resize 512x512 \
images/input.pgm

echo "Conversion completed."
echo "Saved as: images/input.pgm"

# =======================================
# COMPILE PROGRAMS
# =======================================

echo ""
echo "Compiling programs..."

# SERIAL
gcc -fopenmp src/serial_sobel.c \
-o build/serial_sobel -lm

# OPENMP
gcc -fopenmp src/openmp_sobel.c \
-o build/openmp_sobel -lm

# MPI
mpicc src/mpi_sobel.c \
-o build/mpi_sobel -lm

# MPI GHOST
mpicc src/mpighost_sobel.c \
-o build/mpighost_sobel -lm

# SSD
gcc src/ssd_compare.c \
-o build/ssd_compare -lm

echo "Compilation completed."

# =======================================
# RUN SERIAL
# =======================================

echo ""
echo "[1] Running Serial..."

./build/serial_sobel

# =======================================
# RUN OPENMP
# =======================================

echo ""
echo "[2] Running OpenMP..."

export OMP_NUM_THREADS=4

./build/openmp_sobel

# =======================================
# RUN MPI
# =======================================

echo ""
echo "[3] Running MPI..."

mpirun -np 4 ./build/mpi_sobel

# =======================================
# RUN MPI GHOST
# =======================================

echo ""
echo "[4] Running MPI Ghost..."

mpirun -np 4 ./build/mpighost_sobel

# =======================================
# RUN SSD
# =======================================

echo ""
echo "[5] Running SSD Comparison..."

./build/ssd_compare

# =======================================
# RUN BENCHMARK
# =======================================

echo ""
echo "[6] Running Benchmark..."

./scripts/benchmark.sh

echo ""
echo "======================================="
echo " Pipeline Completed Successfully"
echo "======================================="