# Constrained Multiphysics PINN for Piezoelectric Tensor Discovery

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Official code repository for the manuscript:  
**"Inverse Discovery of Shear Piezoelectric Coupling via Constrained Multiphysics Neural Networks"** *(Submitted to Physical Review Applied)*

## Overview
This repository implements a Physics-Informed Neural Network (PINN) designed to solve stiff, strongly coupled electro-elastodynamics. By applying an exact $t^2$ temporal multiplier and a spatial distance function ($\mathcal{D}_{bc}$) directly to the neural network topology, the architecture completely bypasses the initial-state electrostatic singularities ("Poisson explosion") inherent to unconstrained soft-penalty formulations.

The code executes a 2-stage (Adam $\rightarrow$ L-BFGS) inverse discovery pipeline to successfully isolate and discover the dimensionless $e_{15}$ shear piezoelectric parameter of a PZT-5H medium from sparse subsurface acoustic data.

## Repository Structure
- `src/physics.py`: Contains normalization constants, dimensional scaling, and the purely sinusoidal ($n=2$) manufactured wavefields.
- `src/model.py`: Defines the `PiezoInversePINN` architecture featuring exact algebraic boundary/initial constraints.
- `src/main.py`: Orchestrates the PDE residuals, computes the loss, runs the optimization pipeline, and generates publication-ready graphics.

## Requirements
- Python 3.9+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib

## Usage
To train the constrained model and run the inverse parameter discovery from scratch, execute:
```bash
python src/main.py
