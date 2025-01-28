# PyWST: WST and RWST for Astrophysics

**PyWST** is a public Python package designed for performing statistical analyses of two-dimensional data using the **Wavelet Scattering Transform (WST)** and the **Reduced Wavelet Scattering Transform (RWST)**.

The WST/RWST provides a comprehensive set of coefficients that efficiently describe the non-Gaussian features of your data.

Install PyWST and explore our [Jupyter notebook tutorial](examples/tutorial.ipynb) available in the [examples/](examples/) folder.

If you use this package, please cite the following paper:

B. Regaldo-Saint Blancard, F. Levrier, E. Allys, E. Bellomi, F. Boulanger, "Statistical description of dust polarized emission from the diffuse interstellar medium - A RWST approach", [*Astronomy \& Astrophysics*, 642, A217](https://doi.org/10.1051/0004-6361/202038044) (2020). ArXiv: [2007.08242](https://arxiv.org/abs/2007.08242)

*Note:* For GPU-accelerated WST computations, consider using [kymatio](https://github.com/kymatio/kymatio) (which served as an initial inspiration for parts of this code).

## Install/Uninstall

### Standard installation (from PyPI)

Run in a terminal:

```bash
pip install pywst
```

### Install from source

Clone the repository and run the following command from the main directory:

```bash
pip install .
```

### Uninstall

```bash
pip uninstall pywst
```

## Changelog

### 1.0

* Minor updates.

### 0.9

* First public version.
