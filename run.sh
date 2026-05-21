#!/bin/bash

echo "======================================="
echo " Sobel Edge Detector - HPC Project"
echo "======================================="

echo ""
echo "[1] Compiling Serial Version..."
gcc -fopenmp src/serial_sobel.c -o serial -lm

if [ $? -ne 0 ]; then
    echo "Serial compilation failed!"
    exit 1
fi

echo "Serial compilation successful."

echo ""
echo "[2] Running Serial Version..."
./serial

echo ""
echo "---------------------------------------"

echo ""
echo "[3] Compiling OpenMP Version..."
gcc -fopenmp src/omp_sobel.c -o parallel -lm

if [ $? -ne 0 ]; then
    echo "OpenMP compilation failed!"
    exit 1
fi

echo "OpenMP compilation successful."

echo ""
echo "[4] Running OpenMP Version..."
./parallel

echo ""
echo "---------------------------------------"

echo ""
echo "[5] Compiling MPI Version..."
mpicc src/mpi_sobel.c -o mpi_sobel -lm

if [ $? -ne 0 ]; then
    echo "MPI compilation failed!"
    exit 1
fi

echo "MPI compilation successful."

echo ""
echo "[6] Running MPI Version..."
mpirun -np 4 ./mpi_sobel

echo ""
echo "======================================="
echo " All Executions Completed Successfully"
echo "======================================="