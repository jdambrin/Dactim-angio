# Dactim-angio

> ⚠️ **À compléter** : une phrase de description officielle du projet (objectif scientifique, contexte de l'équipe).

`dactim_angio` is a Python research library for **angiographic image processing**, providing
spatial grids, projective geometry, radial basis function interpolation, and differentiable
similarity metrics — building blocks for **2D/3D registration and reconstruction** of vascular imaging.

## Features

- **Spatial grids** (`dactim_angio.spatial.grid`)
  - Abstract `SpatialGrid` with `push`/`pull` coordinate transforms and finite-difference gradients
  - `SpatialGridAffine` — affine-mapped volumes, loadable directly from NIfTI files
  - `SpatialGrid_XA` — X-ray angiography grids built from NIfTI + JSON metadata (projection center, etc.)
  - Gaussian image pyramids, rescaling, scalar-field transfer between grids, and PyVista export
- **Projective geometry** (`dactim_angio.spatial.geom`)
  - `PointCloud` and `RayFan` primitives with KD-tree / BallTree spatial indexing
  - Sparse point–point, ray–ray and point–ray distance matrices (with analytic derivatives)
  - `Camera` model with field-of-view frustum and visibility queries
  - Parametrized affine transforms (rotation/translation) with closed-form derivatives
- **Radial basis function interpolation** (`dactim_angio.spatial.RBF`)
  - `GaussianRBF` over point clouds — evaluation on points or rays, with shape derivatives
  - `GaussianRBF_rays` over angular ray fans, plus L² / H¹₀ norm matrices
- **Differentiable similarity metrics** (`dactim_angio.metrics`)
  - Mutual information (kernel density estimator), Kraskov joint entropy, correlation, MSE
  - Each returns the metric value **and** its gradients, ready for gradient-based optimization

## Installation

> Requires Python ≥ 3.9.

From source:

```bash
git clone https://github.com/jdambrin/Dactim-angio.git
cd Dactim-angio
pip install .
```

Once published on PyPI:

```bash
pip install dactim-angio
```

## Quick start

> The snippet below is **illustrative** and follows the public API — adapt paths and field names to your data.

```python
from dactim_angio.spatial.grid import SpatialGridAffine
from dactim_angio.metrics import mse

# Load a 3D volume from a NIfTI file
grid = SpatialGridAffine.fromNifti("patient.nii.gz")
print(grid)                      # shape + scalar-field summary

# Export to PyVista for 3D visualization
mesh = grid.toPyvista()

# Compute a differentiable similarity metric between two fields
value, grad_x, grad_y = mse(fixed_field, moving_field)
```

## Package structure

```
dactim_angio/
├── metrics.py            # similarity metrics with analytic gradients
└── spatial/
    ├── geom.py           # point clouds, ray fans, cameras, affine transforms
    ├── grid.py           # spatial grids, NIfTI/DICOM I/O, pyramids
    └── RBF.py            # Gaussian RBF interpolation
```

## Dependencies

`numpy`, `scipy`, `matplotlib`, `nibabel`, `pydicom`, `pyvista`, `scikit-image`, `scikit-learn`.
These are installed automatically with the package.

## License

This software is distributed under the **CeCILL Free Software License Agreement v2.1**
(see the [`LICENSE`](LICENSE) file). CeCILL is a copyleft license, GPL-compatible, governed
by French law, authored by CEA, CNRS and Inria.

## Citation

> ⚠️ **Optionnel** : si ce travail accompagne une publication, ajoute ici la référence à citer.
