# Constrained Multiphysics PINN for Piezoelectric Tensor Discovery

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Official code repository for the manuscript:
**"Exact Spatio-Temporal Constraints to Eliminate Initial-State Electrostatic Singularity in Inverse Piezoelectric Tensor Identification"** *(Submitted to Physical Review Applied)*

## Overview
This repository implements a Physics-Informed Neural Network (PINN) designed to solve stiff, strongly coupled electro-elastodynamics. By applying an exact $t^2$ temporal multiplier and a spatial distance function ($\mathcal{D}_{bc}$) directly to the neural network topology, the architecture completely bypasses the initial-state electrostatic discontinuities inherent to unconstrained soft-penalty formulations. 

Coupled with a rigorous asymptotic dimensional stabilization scheme, the code executes a two-stage (Adam $\rightarrow$ L-BFGS) inverse discovery pipeline to successfully isolate and discover the dimensionless $e_{15}$ shear piezoelectric parameter of a PZT-5H medium. The repository includes both the exact Method of Manufactured Solutions (MMS) verification and a discrete numerical cross-validation against noisy COMSOL Multiphysics FE data.

## Repository Structure

### 1. `src/` (Exact MMS Verification)
* `physics.py`: Contains normalization constants, dimensional scaling, and the decoupled, full-wavelength ($n=2$) manufactured wavefields.
* `model.py`: Defines the `PiezoInversePINN` architecture featuring exact algebraic boundary/initial constraints.
* `main.py`: Orchestrates the PDE residuals, computes the loss, and runs the baseline optimization pipeline against the exact synthetic data.

### 2. `data/` (COMSOL FE Wavefields)
* `comsol_pinn_dataset.csv`: The independent transient wavefield dataset extracted from a COMSOL Multiphysics solid mechanics and electrostatics simulation. Contains both the pristine (`_true`) field data and the 30 dB additive Gaussian noise (`_noisy`) target fields.

### 3. `fem_study/` (Cross-Validation & Noise Analysis)
* `fem_inversion.py`: Applies the constrained architecture to the 30 dB noisy COMSOL data to extract $e_{15}$ under realistic measurement and FEM discretization errors.
* `snr_sweep.py`: An automated parameter sweep script that dynamically injects Gaussian noise from 40 dB down to 0 dB (equal signal-to-noise power) using on-the-fly injection to establish the noise-limited accuracy floor of the framework.

### 4. `matlab/` (Publication Graphics)
* `plot_snr_sensitivity.m`: A standalone MATLAB script utilizing the `tex` interpreter to generate the rigorously formatted, logarithmic noise sensitivity degradation curve (Figure 6 in the manuscript).

## Requirements
* Python 3.9+
* PyTorch 2.0+
* NumPy
* Pandas
* SciPy
* Matplotlib
* MATLAB (Optional, for generating Figure 6)

## Usage

**1. To run the exact MMS verification:**
```bash
python src/main.py
