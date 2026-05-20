from pathlib import Path

content = """
# 🐧 Ubuntu 22.04 Setup Guide for OpenMP & MPI

This guide explains how to set up the development environment for the Sobel Edge Detector HPC Project on Ubuntu 22.04.

---

# 📦 Update Ubuntu

Open terminal and run:

sudo apt update
sudo apt upgrade -y

---

# ⚙️ Install Required Packages

Install GCC, OpenMP, MPI, Git, and build tools:

sudo apt install build-essential gcc g++ make git -y

Install MPI:

sudo apt install openmpi-bin openmpi-common libopenmpi-dev -y

---

# ✅ Verify GCC Installation

gcc --version

---

# ✅ Verify MPI Installation

mpicc --version

Check MPI version:

mpirun --version

---

# ✅ Verify OpenMP Support

Compile OpenMP code:

gcc -fopenmp test_openmp.c -o test

Run:

./test

---

# 🚀 Clone Project Repository

git clone https://github.com/anton20001306/Sobel_Edge_Detector_HPC-.git

Enter project folder:

cd Sobel_Edge_Detector_HPC-

Checkout development branch:

git checkout anton-branch

---

# 🧠 Recommended Project Structure

```text
Sobel_Edge_Detector_HPC-/
├── src/
│   ├── serial_sobel.c
│   ├── omp_sobel.c
│   └── mpi_test.c
│
├── images/
│   └── input.jpg
│
├── results/
│   ├── edges.png
│   ├── serial_output.png
│   └── parallel_output.png
│
├── docs/
│   ├── report.docx
│   └── presentation.pptx
│
├── README.md
├── README_SET_UBUNTU_ENV.md
└── run.sh
```

---

# ▶️ Compile Serial Version

gcc -fopenmp src/serial_sobel.c -o serial -lm

Run:

./serial

---

# ▶️ Compile OpenMP Parallel Version

gcc -fopenmp src/omp_sobel.c -o parallel -lm

Run:

./parallel

---

# ▶️ Compile MPI Test Program

mpicc src/mpi_test.c -o mpi_test

Run with 4 processes:

mpirun -np 4 ./mpi_test

---

# 🛠️ Open Project in VS Code

Inside project folder:

code .

---

# 📊 HPC Concepts Used

- Serial Computing
- Parallel Computing
- OpenMP
- MPI
- Race Conditions
- Thread Scheduling
- Execution Time Analysis
- SSD (Sum of Squared Difference)

---

# 🐧 Environment

- OS: Ubuntu 22.04
- Compiler: GCC
- Parallel Library: OpenMP
- Distributed Library: MPI
- IDE: VS Code

---

Happy Coding 🚀
"""

file_path = "/mnt/data/README_SET_UBUNTU_ENV.md"

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("README_SET_UBUNTU_ENV.md created successfully!")
print(file_path)

