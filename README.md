# Custom CUDA Kernels in Python with Numba

This repository contains code samples and exercises developed as part of the NVIDIA Deep Learning Institute training course on Custom CUDA Kernels in Python with Numba. The projects demonstrate how to write and optimize custom CUDA kernels in Python using Numba, covering topics such as grid stride loops, atomic operations, and device-level random number generation.

## Contents

- **A First CUDA Kernel & Tweak the Code**  
  Demonstrates a simple CUDA kernel for element-wise addition, along with experiments modifying the execution configuration (threads per block, blocks per grid, and synchronization).

- **Accelerate a CPU Function as a Custom CUDA Kernel**  
  Implements the `square_device` kernel that squares each element of a NumPy array on the GPU.

- **Grid Stride Loop Implementation**  
  Implements the `hypot_stride` kernel to compute the hypotenuse of elements in an array using grid stride loops.

- **Accelerated Histogramming Kernel**  
  Provides the `cuda_histogram` kernel which uses atomic operations and grid stride loops to compute a histogram from an input dataset.

- **Monte Carlo Pi on the GPU**  
  Uses a CUDA kernel with device random number generators to estimate the value of π using a Monte Carlo simulation.

## Requirements

- Python 3.x  
- [Numba](http://numba.pydata.org/)  
- [NumPy](https://numpy.org/)  
- NVIDIA CUDA-capable GPU with proper CUDA drivers installed

## Getting Started

Open the Jupyter Notebook `Custom CUDA Kernels in Python with Numba.ipynb` to run the interactive exercises.

Alternatively, run individual scripts such as `histogram.py`:

python histogram.py

## Project Structure

📂 Custom-CUDA-Kernels  
│── Custom CUDA Kernels in Python with Numba.ipynb  → Main Jupyter Notebook with explanations & exercises  
│── histogram.py  → Solution for the accelerated histogramming exercise  

## Certification

After completing these exercises, a certificate of completion was awarded by NVIDIA Deep Learning Institute [My Certificate](https://learn.nvidia.com/certificates?id=foEkX986ROGs38OhPaBjvg). This repository documents the work and learning achieved during the course.


## Acknowledgments

- **NVIDIA Deep Learning Institute (DLI)** for the training materials and certification.  
- **The Numba and CUDA communities** for their continuous support and documentation.
