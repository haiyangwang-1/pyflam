PyFLAM
======

PyFLAM is a minimal NumPy/SciPy Python port of the dense core API from
[FLAM](https://github.com/klho/FLAM), focused on `hypoct`, `id`, `rskel`,
`rskelf`, `ifmm`, and their apply/solve helpers.

The API intentionally follows the MATLAB function signatures, with one Python
adaptation: matrix and proxy callbacks receive 0-based NumPy index arrays.

This project is GPL-3.0-or-later because it is a faithful port of GPLv3 FLAM.
