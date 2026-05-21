#!/bin/bash

# Ensure an input image was provided
if [ -z "$1" ]; then
    echo "Usage: ./benchmark_pipeline.sh <path_to_image_file>"
    echo "Example: ./benchmark_pipeline.sh animal.jpeg"
    exit 1
fi

INPUT_IMAGE=$1
echo "=========================================================="
echo "      HPC Sobel Edge Detection - Benchmarking Pipeline    "
echo "=========================================================="

# Step 1: Pre-processing
echo ""
echo "[1/5] Converting '$INPUT_IMAGE' to grayscale PGM..."
# Converts any format to grayscale, removes compression, saves as input.pgm
convert "$INPUT_IMAGE" -colorspace Gray -depth 8 input.pgm
if [ $? -ne 0 ]; then
    echo "Error: ImageMagick conversion failed."
    exit 1
fi

# Step 2: Compilation
echo ""
echo "[2/5] Compiling all implementations..."
gcc src/serial_sobel.c -o serial -lm -fopenmp
gcc -fopenmp src/omp_sobel.c -o parallel -lm
mpicc src/mpi_sobel.c -o mpi_sobel -lm
nvcc src/hybrid_sobel.cu -o hybrid_sobel -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

# Step 3: Execution & Benchmarking
echo ""
echo "[3/5] Executing and capturing execution times..."

echo "------------------------------------------------"
echo "-> Running Serial Baseline:"
./serial

echo "------------------------------------------------"
echo "-> Running OpenMP (Shared Memory):"
for threads in 1 2 4 8; do
    echo "   [ Testing with $threads Threads ]"
    export OMP_NUM_THREADS=$threads
    ./parallel
done

echo "------------------------------------------------"
echo "-> Running MPI (Distributed Memory):"
for procs in 1 2 4; do
    echo "   [ Testing with $procs Processes ]"
    mpirun -np $procs --mca btl ^vader ./mpi_sobel
done

echo "------------------------------------------------"
echo "-> Running Hybrid (MPI + CUDA GPU):"
for procs in 1 2 4; do
    echo "   [ Testing with $procs Processes & GPU ]"
    mpirun -np $procs --mca btl ^vader ./hybrid_sobel
done

# Step 4: Accuracy Validation (RMSE)
echo ""
echo "------------------------------------------------"
echo "[4/5] Calculating Accuracy (RMSE) against Serial Baseline..."
echo "(Scores closer to 0 mean higher mathematical accuracy)"

# ImageMagick calculates RMSE and outputs to stderr, so we append 2>&1 to capture it
OMP_RMSE=$(compare -metric RMSE serial_output.pgm omp_output.pgm null: 2>&1)
MPI_RMSE=$(compare -metric RMSE serial_output.pgm mpi_output.pgm null: 2>&1)
HYBRID_RMSE=$(compare -metric RMSE serial_output.pgm hybrid_output.pgm null: 2>&1)

echo "OpenMP RMSE : $OMP_RMSE"
echo "MPI RMSE    : $MPI_RMSE"
echo "Hybrid RMSE : $HYBRID_RMSE"

# Step 5: Output Conversion
echo ""
echo "[5/5] Generating final PNG outputs..."
convert serial_output.pgm final_serial_edges.png
convert omp_output.pgm final_omp_edges.png
convert mpi_output.pgm final_mpi_edges.png
convert hybrid_output.pgm final_hybrid_edges.png

echo ""
echo "=========================================================="
echo " Pipeline Complete! "
echo " You can now use the terminal data for your Analysis Report."
echo " The final processed images are saved as .png files."
echo "=========================================================="