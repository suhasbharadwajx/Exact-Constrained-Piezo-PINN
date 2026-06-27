# Exact Spatio-Temporal Constrained PINN for Piezoelectric Tensor Discovery

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Official code repository for the manuscript:
**"Exact Spatio-Temporal Constraints to Eliminate Initial-State Electrostatic Inconsistencies in Inverse Piezoelectric Tensor Identification"** *(Submitted to Physical Review Applied)*

## Overview
This repository implements a Physics-Informed Neural Network (PINN) designed to solve stiff, strongly coupled electro-elastodynamics. By applying an exact $t^2$ temporal multiplier and a spatial distance function ($\mathcal{D}_{bc}$) directly to the neural network topology, the architecture completely bypasses the initial-state electrostatic inconsistencies inherent to unconstrained soft-penalty formulations. 

Coupled with a rigorous asymptotic dimensional stabilization scheme, the code executes a two-stage (Adam $\rightarrow$ L-BFGS) inverse discovery pipeline to successfully isolate the dimensionless $e_{15}$ shear piezoelectric parameter of a PZT-5H medium. The repository is modularized into two primary frameworks: an exact Method of Manufactured Solutions (MMS) verification, and a discrete numerical cross-validation against noisy COMSOL Multiphysics FE data. To rigorously prevent deceptive optimization shortcuts, both studies evaluate the network against a highly asymmetric, 8-lobe non-polynomial $[1-\cos(2\pi t)]$ transient wavefield.

## Repository Structure

### 1. `MMS-framework/` (Exact MMS Verification)
This directory isolates the network validation against the mathematically exact, noise-free analytical wavefield.
* **`physics.py`**: Contains normalization constants, dimensional scaling, and the exact asymmetric 8-lobe continuous analytical body-force equations.
* **`model.py`**: Defines the `ConstrainedPINN` architecture featuring exact algebraic boundary and initial constraints.
* **`main.py`**: Orchestrates the PDE residuals, computes the loss, and runs the optimization pipeline against the exact synthetic data.

### 2. `FEM-framework/` (Cross-Validation & Noise Analysis)
This directory applies the identical constrained architecture to discrete, full-field finite element data to evaluate performance under realistic measurement and discretization errors.
* **`physics.py` & `model.py`**: The identical continuum physics mapping and constrained neural architecture.
* **`main.py`**: The master orchestration script that loads the FE data, dynamically injects 30 dB Gaussian noise, trains the network, and discovers the $e_{15}$ parameter.
* **`SNR-sweep.py`**: The execution script for the comprehensive noise sensitivity study, running $N=5$ independent stochastic initializations across 9 discrete SNR levels (40 dB down to 0 dB). *(Note: Even with optimized batch sizes and reduced epochs, this comprehensive sweep is computationally intensive and requires approximately 28 hours to complete on a standard NVIDIA Tesla T4 GPU).*
* **`Plot-SNR-sweep.py`**: Generates the final shaded variance band publication graphics (Figure 6) from the sweep results.

### 3. `data/`
* **`FEM-Dataset.txt.zip`**: The uncorrupted transient wavefield dataset extracted directly from the COMSOL Multiphysics solid mechanics and electrostatics simulation. *(Note: Unzip this file into the working directory prior to running the FEM scripts).*

## Requirements
* Python 3.9+
* PyTorch 2.0+ (Utilizes TF32 hardware acceleration)
* NumPy
* Pandas
* Matplotlib

## Usage

**1. To run the exact MMS verification:**
```bash
python MMS-framework/main.py
