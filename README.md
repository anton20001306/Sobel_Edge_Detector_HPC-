# Parallel Sobel Edge Detector

**EE7218 – High Performance Computing | Group 02**  
Faculty of Engineering, University of Ruhuna

> A high-performance Sobel edge detection pipeline implemented in four progressively parallel variants — Serial, OpenMP, MPI, and MPI+CUDA Hybrid — with a fully automated benchmark pipeline and report-ready result generation.

---

## Table of Contents

- [Overview](#overview)
- [Implementations](#implementations)
- [Results](#results)
- [Sample Output](#sample-output)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Running Individual Implementations](#running-individual-implementations)
- [Generating Plots](#generating-plots)
- [How It Works](#how-it-works)
- [Authors](#authors)

---

## Overview

Edge detection is a fundamental operation in image processing. This project implements the **Sobel operator** — a two-stage convolution pipeline — across four parallelisation paradigms and benchmarks their performance on a 5600 × 3200 pixel (17.92 MP) greyscale image.

The pipeline for all implementations:

```
Load PGM  →  Gaussian Blur (3×3)  →  Sobel Gradient (Gx, Gy)  →  Compute |G|  →  Save PGM
```

Gaussian smoothing is applied before edge detection to suppress noise. Every parallel implementation produces **byte-identical output** to the serial reference (RMSE = 0.0000), confirming correctness at every configuration.

---

## Implementations

| File | Method | Parallelism Model |
|---|---|---|
| `serial_sobel.c` | Serial baseline | Single thread, sequential |
| `omp_sobel.c` | OpenMP | Shared-memory, row-slab decomposition |
| `mpi_sobel.c` | MPI | Distributed-memory, ghost-row exchange |
| `hybrid_sobel.cu` | MPI + CUDA | Distributed decomposition + GPU offload |

### Serial
Straightforward nested loop over all interior pixels. Timed with `omp_get_wtime()` (I/O excluded). Serves as the speedup reference for all other implementations.

### OpenMP
The outer row loop of each convolution stage is parallelised with `#pragma omp for schedule(static, chunk)` inside a single `#pragma omp parallel` region. An implicit barrier between the Gaussian and Sobel stages ensures correctness. Tested with 1, 2, 4, 8, and 16 threads.

### MPI
Row slabs are distributed to all ranks via `MPI_Scatterv`. Each rank allocates `(local_rows + 2) × width` pixels — two ghost rows that hold boundary data from neighbouring ranks, exchanged with `MPI_Sendrecv` before each convolution stage. Results are collected with `MPI_Gatherv`. Timing uses `MPI_Reduce(MPI_MAX)` across all ranks. Tested with 1, 2, 4, and 8 processes.

### MPI + CUDA Hybrid
Each MPI rank receives its row slab via `MPI_Scatterv`, transfers it to GPU device memory (`cudaMemcpy` H2D), and launches 2-D CUDA kernel grids (16×16 thread blocks) for both the Gaussian and Sobel passes. Ghost-row exchange on the blur buffer is performed between kernels on the host. Results are transferred back (`cudaMemcpy` D2H) and gathered to Rank 0. GPU transfer times are recorded separately via `cudaEvent_t`. Tested with 4 MPI ranks.

---

## Results

All benchmarks run on: **8-core CPU host + CUDA-capable GPU**, image size **5600 × 3200 px (17.92 MP)**, median of 3 runs, I/O excluded from timing.

| Implementation | Workers | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| Serial (baseline) | 1 thread | 0.2666 | 1.00× | 100.0% |
| OpenMP | 1 thread | 0.2549 | 1.05× | 104.6% |
| OpenMP | 2 threads | 0.1324 | 2.01× | 100.7% |
| OpenMP | 4 threads | 0.1391 | 1.92× | 47.9% |
| OpenMP | 8 threads | 0.1138 | 2.34× | 29.3% |
| OpenMP | 16 threads | 0.1110 | 2.40× | 15.0% |
| MPI | 1 process | 0.2553 | 1.04× | 104.4% |
| MPI | 2 processes | 0.1214 | 2.20× | 109.8% |
| MPI | 4 processes | 0.0870 | 3.06× | 76.6% |
| MPI | 8 processes | 0.0587 | 4.54× | 56.7% |
| **MPI + CUDA Hybrid** | **4 ranks** | **0.0337** | **7.91×** | — |

**GPU transfer overhead:** H2D = 7.8 ms · D2H = 8.3 ms · Total transfer ≈ 48% of hybrid runtime

**Correctness:** All outputs pass RMSE = 0.0000 vs serial baseline.

---

## Sample Output

| Original Image | Edge Detection Output |
|---|---|
| ![Input](Serial/docs/animal.jpeg) | ![Output](Serial/docs/output_hybrid.png) |

> Place your sample images in a `docs/` folder and update the paths above.  
> Run the pipeline once and copy `pipeline_output/input.pgm` (convert to JPG) and any `output_*.pgm` (convert to PNG) into `docs/`.

---

## Project Structure

```
.
├── serial_sobel.c        # Serial baseline implementation
├── omp_sobel.c           # OpenMP shared-memory implementation
├── mpi_sobel.c           # MPI distributed-memory implementation
├── hybrid_sobel.cu       # MPI + CUDA hybrid implementation
├── run_pipeline.sh       # Full automated benchmark pipeline
├── run_multi_res.sh      # Multi-resolution benchmark wrapper
├── make_plots.py         # Generates comparison plots from results
├── results_report.txt    # Latest benchmark results
└── docs/                 # Images for README (add your own)
```

---

## Requirements

### All implementations
```bash
sudo apt-get install gcc libopenmpi-dev openmpi-bin imagemagick python3 python3-pip
pip install numpy matplotlib --break-system-packages
```

### CUDA hybrid only
- NVIDIA CUDA Toolkit (`nvcc`)
- CUDA-capable GPU
- OpenMPI built with CUDA support (GPU-Direct not required)

Verify your setup:
```bash
gcc --version
mpicc --version
nvcc --version          # optional, for hybrid only
nvidia-smi              # optional, for hybrid only
```

---

## Quick Start

Clone the repo and run the full benchmark pipeline on any image:

```bash
git clone <your-repo-url>
cd <repo-folder>
chmod +x run_pipeline.sh
./run_pipeline.sh animal.jpeg
```

This will automatically:
1. Check all dependencies
2. Convert your image to greyscale PGM
3. Compile all four implementations
4. Run serial, OpenMP (×5 thread counts), MPI (×4 process counts), and hybrid
5. Validate correctness (RMSE vs serial) for every output
6. Print a formatted results table
7. Save everything to `results_report.txt`
8. Convert all PGM outputs to PNG for easy viewing

**Accepts any image format** — JPG, PNG, BMP, TIFF, PGM. ImageMagick handles the conversion automatically.

---

## Running Individual Implementations

### Compile everything manually

```bash
# Serial
gcc -O2 -fopenmp -o serial_sobel serial_sobel.c -lm

# OpenMP
gcc -O2 -fopenmp -o omp_sobel omp_sobel.c -lm

# MPI
mpicc -O2 -o mpi_sobel mpi_sobel.c -lm

# MPI + CUDA hybrid
nvcc -O2 hybrid_sobel.cu $(mpicc --showme:compile) $(mpicc --showme:link) -lm -o hybrid_sobel
```

### Run each implementation

```bash
# Serial
./serial_sobel input.pgm output_serial.pgm

# OpenMP — specify thread count as third argument
./omp_sobel input.pgm output_omp.pgm 8

# MPI — specify process count with -np
mpirun --oversubscribe -np 4 ./mpi_sobel input.pgm output_mpi.pgm

# Hybrid — 4 MPI ranks, each uses GPU
mpirun --oversubscribe -np 4 ./hybrid_sobel input.pgm output_hybrid.pgm
```

> **Note:** Input must be a binary PGM (P5) file. Use the pipeline script or ImageMagick to convert:
> ```bash
> convert your_image.jpg -colorspace Gray -type Grayscale input.pgm
> ```

---

## Generating Plots

After running the pipeline, generate comparison charts with:

```bash
python3 make_plots.py results_report.txt
```

For a multi-resolution comparison across three image sizes (25%, 50%, 100% of original):

```bash
chmod +x run_multi_res.sh
./run_multi_res.sh animal.jpeg
```

This produces five report-ready plots in `plots/`:

| File | Description |
|---|---|
| `01_speedup_vs_workers.png` | Speedup curves — OpenMP vs MPI vs hybrid |
| `02_efficiency_vs_workers.png` | Parallel efficiency (%) vs worker count |
| `03_time_vs_image_size.png` | Execution time vs image size (log-log) |
| `04_best_speedup_vs_size.png` | Best speedup per implementation vs image size |
| `05_implementation_comparison.png` | Grouped bar chart of best time per implementation |

---

## How It Works

### Sobel Operator

The algorithm applies two 3×3 convolution kernels to approximate horizontal (Gx) and vertical (Gy) intensity gradients, then combines them:

```
        [-1  0  1]          [-1  -2  -1]
  Gx =  [-2  0  2]    Gy =  [ 0   0   0]
        [-1  0  1]          [ 1   2   1]

  |G| = sqrt(Gx² + Gy²),  clamped to [0, 255]
```

A Gaussian blur (3×3, sum=16) is applied first to suppress noise.

### Parallelisation Strategy

**OpenMP** splits the image row range across threads. All threads share the same image, blur, and edge buffers — they write to non-overlapping rows, so no locks are needed. An implicit barrier after the blur pass synchronises threads before the Sobel pass begins.

**MPI** gives each rank a slab of rows. Because the 3×3 kernels need one row of context from neighbouring ranks, a one-pixel ghost row is exchanged above and below each slab using `MPI_Sendrecv` before each convolution stage.

**MPI+CUDA** uses the same row-slab decomposition as MPI, but within each rank the convolution is offloaded to the GPU. The CPU manages data movement (H2D, ghost exchange, D2H) while the GPU performs thousands of pixel computations in parallel using a 16×16 thread-block grid.

---

## Authors

**Group 02 — Electrical & Information Engineering**  
Faculty of Engineering, University of Ruhuna

| Registration | Name |
|---|---|
| EG/2021/4512 | W.A.P.N Fernando |
| EG/2021/4654 | S.W.M Madhusan |

**Course:** EE7218 – High Performance Computing

---

*This project is submitted for academic purposes. No license is granted for reuse or redistribution.*
