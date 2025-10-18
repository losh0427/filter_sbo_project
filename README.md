# Cavity Filter Bayesian Optimization Framework

## Overview

This project implements a **Bayesian Optimization framework** for optimizing cavity filter designs using HFSS electromagnetic simulations. The system uses Gaussian Process (GP) surrogate models to efficiently explore a 10-dimensional parameter space, minimizing the number of expensive HFSS simulations required.

## What This Project Does

The framework optimizes trapezoidal cavity filters with groove patterns by:
1. Using **Gaussian Process models** (BoTorch/GPyTorch) to learn the relationship between design parameters and filter performance
2. Employing **acquisition functions** (EI/UCB/MES) to intelligently select next design candidates
3. Automating **HFSS simulations** to evaluate electromagnetic field distributions
4. Iteratively improving designs to maximize electric field intensity at the cavity exit

**Key Goal**: Find optimal cavity filter geometries with minimal simulation runs through intelligent sampling.


**Parameter Space** (10D):
- Cavity geometry: top_width, bottom_width, height, length
- Groove pattern: tooth_depth, tooth_length, tooth_width, groove_length, groove_width, extension_factor

## Technology Stack

- **BoTorch/GPyTorch**: Gaussian Process surrogate modeling
- **PyTorch**: Tensor operations and optimization
- **Ansys HFSS**: Electromagnetic field simulations
- **NumPy**: Scientific computing

## Project Structure

```
filter_sbo_project/
├── scripts/
│   ├── functions.py          # Utility functions (I/O, parameter management)
│   ├── models.py             # GP models and acquisition functions
│   ├── pipeline.py           # Main optimization loop
│   ├── hfss_script.py        # HFSS automation
│   ├── initial.py            # Initial sampling
│   └── test_*.py             # Testing utilities
├── Data/                     # Archive (dataset.txt)
└── Data1/                    # Working directory (input/output files)
```

## Quick Start

### 1. Initial Sampling
Generate random samples and run HFSS simulations:
```bash
cd scripts
python initial.py --generate --n_samples 300 --data_dir ../Data1
# Run HFSS simulations
python initial.py --integrate --data_dir ../Data1
```

### 2. Run Optimization
Start the Bayesian optimization pipeline:
```bash
python pipeline.py --path /path/to/project/ --max_iterations 300 --acquisition_func EI
```

## Core Components

- **`functions.py`**: Parameter bounds, file I/O, FLD parsing, objective calculation
- **`models.py`**: GP surrogate model (Matérn kernel), acquisition functions (EI/UCB/MES)
- **`pipeline.py`**: Bayesian optimization loop with immediate file processing
- **`hfss_script.py`**: Automated geometry creation and field export

## References

- **BoTorch**: https://botorch.org/
- **GPyTorch**: https://gpytorch.ai/
- **AutoEncoder Implementation**: https://github.com/esrj/auto_encoder (for future input augmentation)


