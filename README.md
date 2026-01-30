# pyaceqd
Calculate the population dynamics in a QD using ACE, optionally including the interaction with environments (phonons, photons)

See [ACE](https://github.com/mcygorek/ACE) and its [documentation](https://htmlpreview.github.io/?https://github.com/mcygorek/ACE/blob/master/documentation/documentation.html)

Also features the calculation of several multi-time correlation functions.

This assumes that the ACE binary is somewhere in the $PATH.

## Install (editable)

```bash
pip install meson-python meson ninja
PATH=$HOME/.venv/bin:$PATH pip install -e . --no-build-isolation
```

Optional Fortran extensions are built via f2py during install. If the Fortran
build fails, the package still installs and the accelerated routines are
unavailable.

Before running, you might need to add the following to your shell environment:
```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=16
export OMP_STACKSIZE=512M
```
