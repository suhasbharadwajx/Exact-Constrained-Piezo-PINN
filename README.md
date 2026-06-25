# Exact Spatio-Temporal Constrainted PINN for Piezoelectric Tensor Discovery

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Official code repository for the manuscript:
**"Exact Spatio-Temporal Constraints to Eliminate Initial-State Electrostatic Inconsistencies in Inverse Piezoelectric Tensor Identification"** *(Submitted to Physical Review Applied)*

## Overview
This repository implements a Physics-Informed Neural Network (PINN) designed to solve stiff, strongly coupled electro-elastodynamics. By applying an exact $t^2$ temporal multiplier and a spatial distance function ($\mathcal{D}_{bc}$) directly to the neural network topology, the architecture completely bypasses the initial-state electrostatic inconsistencies inherent to unconstrained soft-penalty formulations. 

Coupled with a rigorous asymptotic dimensional stabilization scheme, the code executes a two-stage (Adam $\rightarrow$ L-BFGS) inverse discovery pipeline to successfully isolate and discover the dimensionless $e_{15}$ shear piezoelectric parameter of a PZT-5H medium. The repository is modularized into two primary frameworks: an exact Method of Manufactured Solutions (MMS) verification, and a discrete numerical cross-validation against noisy COMSOL Multiphysics FE data.

## Repository Structure

### 1. `MMS-framework/` (Exact MMS Verification)
This directory isolates the network validation against the mathematically exact, noise-free analytical wavefield.
* **`physics.py`**: Contains normalization constants, dimensional scaling, and the decoupled, full-wavelength ($n=2$) manufactured wavefields.
* **`model.py`**: Defines the `PiezoInversePINN` architecture featuring exact algebraic boundary and initial constraints.
* **`main.py`**: Orchestrates the PDE residuals, computes the loss, and runs the optimization pipeline against the exact synthetic data.

### 2. `FEM-framework/` (Cross-Validation & Noise Analysis)
This directory applies the identical constrained architecture to discrete, full-field finite element data to evaluate performance under realistic measurement and discretization errors.
* **`physics.py`**: Contains the material physics and exact analytical body-force equations applied as volumetric source terms in COMSOL.
* **`model.py`**: The identical constrained neural architecture.
* **`main.py`**: The master orchestration script that loads the FE data, trains the network against 30 dB noisy sensors, and calculates the true $L_2$ reconstruction error.
* **`Raw-FEM-Dataset.csv`**: The pristine, uncorrupted transient wavefield dataset extracted directly from the COMSOL Multiphysics solid mechanics and electrostatics simulation.
* **`Noisy-FEM-Dataset.csv`**: The target training dataset containing the COMSOL wavefield with 30 dB additive Gaussian noise superimposed on all sensor targets.

## Requirements
* Python 3.9+
* PyTorch 2.0+
* NumPy
* Pandas
* SciPy
* Matplotlib
* MATLAB *(Optional, for generating Figure 6)*

## Usage

**1. To run the exact MMS verification:**
```bash
python MMS-framework/main.py
```

**2. To run the FEM Cross-Validation (30 dB Noise):**
```bash
python FEM-framework/main.py
```
