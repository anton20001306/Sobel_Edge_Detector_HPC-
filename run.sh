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
echo "[3] Compiling Parallel Version..."
gcc -fopenmp src/omp_sobel.c -o parallel -lm

if [ $? -ne 0 ]; then
    echo "Parallel compilation failed!"
    exit 1
fi

echo "Parallel compilation successful."

echo ""
echo "[4] Running Parallel Version..."
./parallel

echo ""
echo "======================================="
echo " Execution Completed Successfully"
echo "======================================="
