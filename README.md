# Sobel Edge Detector using OpenMP

This project implements a **Sobel Edge Detector** using both:

- Serial Processing
- Parallel Processing with OpenMP

The project is developed on **Ubuntu 22.04** for HPC (High Performance Computing) learning purposes.

---

# 📌 Project Objectives

- Implement Sobel Edge Detection
- Compare Serial vs Parallel execution
- Analyze race conditions in parallel computing
- Measure execution time
- Calculate SSD (Sum of Squared Difference)
- Study OpenMP performance

---

# 🖼️ Image Processing Pipeline

1. Load input image
2. Convert RGB image to grayscale
3. Apply Gaussian smoothing
4. Apply Sobel operator
5. Compute gradient magnitude
6. Apply thresholding
7. Normalize output image
8. Save edge-detected image

---

# 📂 Project Structure

```text
Sobel_Edge_Detector_HPC-/
│
├── src/
│   ├── serial_sobel.c
│   └── omp_sobel.c
│
├── images/
│   └── input.jpg
│
├── results/
│   ├── edges.png
│   ├── edges.jpg
│   └── edges.pgm
│
├── docs/
│
├── README.md
└── README_SET_UBUNTU_ENV.md

# ⚙️ Build and Run Commands

## ▶️ Serial Version

Compile:

```bash
gcc -fopenmp src/serial_sobel.c -o serial -lm
```

Run:

```bash
./serial
```

---

## ▶️ OpenMP Parallel Version

Compile:

```bash
gcc -fopenmp src/omp_sobel.c -o parallel -lm
```

Run:

```bash
./parallel
```

---

## ▶️ MPI Parallel Version

Compile:

```bash
mpicc src/mpi_sobel.c -o mpi_sobel -lm
```

Run with 4 processes:

```bash
mpirun -np 4 ./mpi_sobel
```
