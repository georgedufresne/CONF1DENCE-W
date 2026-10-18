# CONF1DENCE

**C**Onditional **N**ormalizing **F**low based **1D** stellar **E**volutio**N** a**C**C**E**lerator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fast, flexible stellar population synthesis tool powered by machine learning. CONF1DENCE uses conditional normalizing flows to emulate 1D stellar evolution models, enabling rapid generation of synthetic stellar populations with realistic observational effects.

## 🌟 Features

- **⚡ Fast**: Generate 10,000-50,000 stellar models per second on CPU
- **🎯 Accurate**: Matches traditional stellar evolution models with high fidelity
- **🔧 Flexible**: Support for multiple photometric systems and stellar evolution grids
- **🎨 Complete**: Includes photometric uncertainties, completeness, and binary stars
- **📊 Easy to Use**: Simple API with Jupyter notebook examples

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/200k33p3r/CONF1DENCE.git
cd CONF1DENCE

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
import CONF1DENCE
from CONF1DENCE.predict import predict
import numpy as np

# Initialize the emulator
emulator = predict.emulator()

# Generate 10,000 stars for a globular cluster
n_stars = 10000
ages = np.full(n_stars, 11.8e9)  # 11.8 Gyr
masses = np.random.uniform(0.2, 1.0, n_stars)  # Solar masses

# Set stellar population parameters (metallicity, mixing length, etc.)
MC_params = [-0.77, 0.250, 0.40, ...]  # See examples for full list

# Generate theoretical stellar models
theory_models = emulator.gen_theory_models(labels)

# Generate synthetic Color-Magnitude Diagram with observational effects
mock_CMD = emulator.gen_sCMD(
    MC_params, 
    ages, 
    n_stars,
    pdmf=-0.84,
    radius_dist=your_radius_function,
    distance=13.38,
    bf=0.046  # Binary fraction
)
```

See the [examples](examples/) directory for complete tutorials.

## 📚 Documentation

### Core Components

CONF1DENCE has three main modules:

1. **Stellar Evolution Model (SEM)**: Emulates theoretical stellar evolution using normalizing flows
2. **Bolometric Correction (BC)**: Converts theoretical parameters to observed magnitudes
3. **Artificial Star Tests (AS)**: Applies photometric uncertainties and completeness

### Key Classes

- `predict.emulator()`: Main class for generating stellar populations
- `train.TrainMod()`: Train stellar evolution model emulators
- `train.TrainBC()`: Train bolometric correction tables
- `train.TrainAS()`: Train photometric uncertainty models

## 📖 Examples

We provide comprehensive Jupyter notebooks:

- **[sCMD_construction.ipynb](examples/sCMD_construction.ipynb)**: Complete tutorial for generating synthetic CMDs
- **[quick_start.ipynb](examples/quick_start.ipynb)**: Quick introduction to basic features
- **[Train_BC_tables.ipynb](examples/Train_BC_tables.ipynb)**: How to train custom bolometric correction tables

## 🎯 Performance

Typical performance on CPU (AMD/Intel):
- **Isochrone generation**: ~10,000-50,000 stars/second
- **Full CMD (with observational effects)**: ~5,000-20,000 stars/second
- **Memory usage**: ~100-500 MB for 100,000 stars

GPU acceleration available if PyTorch CUDA is installed.

## 🔬 Scientific Applications

CONF1DENCE is designed for:

- Globular cluster population synthesis
- Stellar population parameter inference
- Mock observation generation for survey planning
- Testing photometric analysis pipelines
- Bayesian inference with MCMC/nested sampling

## 📦 Package Structure

```
CONF1DENCE/
├── CONF1DENCE/          # Main package
│   ├── predict/         # Prediction/emulation module
│   ├── train/           # Training module
│   └── utils/           # Utility functions
├── data/                # Pre-trained models and format files
│   ├── SEM/            # Stellar evolution models
│   ├── BC_table/       # Bolometric correction tables
│   ├── AS_test/        # Artificial star test models
│   └── format/         # JSON format specifications
└── examples/           # Jupyter notebook tutorials
```

## 🛠️ Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12.0
- NumPy ≥ 1.20.0
- pandas ≥ 1.3.0
- zuko ≥ 1.0.0 (normalizing flows)
- h5py ≥ 3.0.0
- matplotlib ≥ 3.4.0
- psutil ≥ 5.8.0


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Martin Ying**  
Email: martin.ying.gr@dartmouth.edu  and george.e.dufresne.gr@dartmouth.edu
GitHub: [@200k33p3r](https://github.com/200k33p3r)

## 🙏 Acknowledgments

- Stellar evolution grids: [DSEP](https://rcweb.dartmouth.edu/stellar/)
- Bolometric corrections: [YBC Database](https://gitlab.com/cycyustc/ybc_tables)
- Normalizing flows: [Zuko](https://github.com/probabilists/zuko)

## 📚 Citation

If you use CONF1DENCE in your research, please cite:

```bibtex
@software{conf1dence,
  author = {Ying, Martin},
  title = {CONF1DENCE: COnditional Normalizing Flow based 1D stellar EvolutioN acCElerator},
  year = {2025},
  url = {https://github.com/200k33p3r/CONF1DENCE}
}
```

## 🗺️ Roadmap

Future features planned:
- [ ] Support for more photometric systems (JWST, Gaia, etc.)
- [ ] Web interface for quick calculations
- [ ] Inference Method

---

