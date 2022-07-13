# aetherpy
[![Coverage Status](https://coveralls.io/repos/github/AetherModel/aetherpy/badge.svg?branch=main)](https://coveralls.io/github/AetherModel/aetherpy?branch=main)

The Python package that supports Aether model data management and analysis.

# Installation

## Starting from scratch
* Python and the packages aetherpy depends on are freely available. Python may
  be obtained through various package managers or from https://www.python.org/.
  A common OS-independent Python package manager to obtain the aetherpy
  dependencies is [PyPi](https://pypi.org/).

## Installation from GitHub

```
git clone https://github.com/AetherModel/aetherpy
cd aetherpy
python setup.py install
```

If you want to install aetherpy for the entire system, you will need to add
`sudo` in front of the final command in the above block.  To install for
yourself without root priveledges, append `--user` to the end of the final
command.

## Installation from PyPi

Pip installation will soon be available

## Installation for development

If you intend to contribute to the testing or development of aetherpy, you will
need to install in develop mode:

```
python setup.py develop
```

This allows you to change branches to the desired test brach or alter files
without needing to re-install aetherpy.

## Installation for development using Conda

For developers, aetherpy's dependencies may be installed using the conda package manager via the following steps:

1. If not already installed, install conda through your preferred distribution of Anaconda / Miniconda.
2. Create a new conda environment with the desired dependencies by running the following command:
```
conda create -n aetherpy_env matplotlib numpy pytest netCDF4 cartopy
```
3. Activate the conda environment and install aetherpy in develop mode:
```
conda activate aetherpy_env
python setup.py develop
```

# Getting Started

aetherpy contains a test script to plot model results.  It may be called via:

```
python aetherpy/run_plot_model_results.py var=3 -alt=120 [file_dir]/3DALL*.bin
```

where [file_dir] is the directory where you have Aether model output files. You
can see the help by typing:

```
python aetherpy/run_plot_model_results.py -h
```
