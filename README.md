# PyWST : WST and RWST for astrophysics

PyWST is a public Python package designed to perform statistical analyses of two-dimensional data with the Wavelet Scattering Transform (WST) and the Reduced Wavelet Scattering Transform (RWST).

The WST/RWST give you convenient sets of coefficients that describe your non-Gaussian data in a comprehensive way.

Install PyWST and check out our [Jupyter notebook tutorial](examples/tutorial.ipynb) in the *examples/* folder.

If you use this package, please cite the following paper:

B. Regaldo-Saint Blancard, F. Levrier, E. Allys, E. Bellomi, F. Boulanger (2020). Statistical description of dust polarized emission from the diffuse interstellar medium - A RWST approach. arXiv preprint [arXiv:2007.08242](https://arxiv.org/abs/2007.08242)

*Note:* For GPU-accelerated WST computations, take a look at [kymatio](https://github.com/kymatio/kymatio) (on which part of this code is based).

## Install/Uninstall

### Standard installation (from the Python Package Index)

Type in a console:

```
pip install pywst
```

### Install from source

Clone the repository and type from the main directory:

```
pip install -r requirements.txt
pip install .
```

### Uninstall

```
pip uninstall pywst
```

## Changelog

### 1.0

* Minor updates.

### 0.9

* First public version.
