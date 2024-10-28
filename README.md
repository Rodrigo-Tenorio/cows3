# Continuous-wave search sensitivity simulator (cows3)

[![PyPI version](https://badge.fury.io/py/cows3.svg)](https://badge.fury.io/py/cows3)
[![arXiv](https://img.shields.io/badge/arXiv-2405.18934-b31b1b.svg)](https://arxiv.org/abs/2405.18934)

A Python package to estimate the sensitivity of general
continuous gravitational-wave searches.

The method should be equivalent to the semi-analytical approach derived in
[Dreissigacker, Prix, Wette (2018)](https://arxiv.org/abs/1808.02459) and
implemented in [Octapps](https://github.com/octapps/octapps), but here we 
implement it in Python to make it more convenient to use.

## How to install 

`cows3` is available under PyPI:

```
pip install cows3
```


## Citing this work

If `cows3` was useful to your research, we would appreciate if you cited
[Mirasola & Tenorio (2024)](https://arxiv.org/abs/2405.18934) where this
implementation was first presented:
```
@article{Mirasola:2024lcq,
    author = "Mirasola, Lorenzo and Tenorio, Rodrigo",
    title = "{Towards a computationally-efficient follow-up pipeline for blind continuous gravitational-wave searches}",
    eprint = "2405.18934",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "LIGO-P2400221",
    month = "5",
    year = "2024",
    journal = "arXiv e-prints"
}
```
as well as a Zenodo release of this software.

For the semi-analytical sensitivity estimation method you should also cite 
[Wette (2012)](https://arxiv.org/abs/1111.5650) and
[Dreissigacker, Prix, Wette (2018)](https://arxiv.org/abs/1808.02459). Also,
this package makes extensive use of SWIG bindings, so please cite
[Wette (2021)](https://arxiv.org/abs/2012.09552) as well.


## Authors
- Rodrigo Tenorio
- Lorenzo Mirasola


