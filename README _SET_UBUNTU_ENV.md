# 🚀 OpenMP Setup Guide (Ubuntu 22.04)

This project uses **OpenMP** for parallel programming in C on Ubuntu 22.04.

---

## 📦 Install Required Tools

Open terminal and run:

```bash
sudo apt update
sudo apt install build-essential gcc g++ make -y
```

---

## ⚙️ Compile OpenMP Program

Use the `-fopenmp` flag when compiling:

```bash
gcc -fopenmp filename.c -o output
```

Example:

```bash
gcc -fopenmp sobel_serial.c -o sobel
```

---

## ▶️ Run the Program

```bash
./output
```

Example:

```bash
./sobel
```

---

## ✅ Verify GCC Installation

```bash
gcc --version
```

---

## 🧠 Check CPU Core Count

```bash
nproc
```

---

## 🐧 Environment

- OS: Ubuntu 22.04
- Compiler: GCC
- Parallel Library: OpenMP

---

Happy Coding 🚀
