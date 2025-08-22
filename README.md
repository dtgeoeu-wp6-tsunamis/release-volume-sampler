# Earthquake Induced Recursive Release Volume Sampler
This repository contains the code developped as part of the *Interoperable earthquake induced landslide module*. The code is designed to generate a database of potential submarine release volumes triggered by earthquakes. Once an ensemble of shakemaps are given, release probabilities are assigned to each volume in the database. To achieve the sought functionality, the operation is divided in two steps.

**Preparational submodule** 
The preparational step creates a database of release volumes. This encompass the following steps:
1. Slope Analysis: Calculation of factor of safety and yield accelleration in a probabilistic framework. 
1. Triangulation: Creation of a triangular mesh used as a spatial discretization for the selection of release volumes. The mesh is fitted using TensorFlow.
1. Volume Sampling: Recursive sampling proceedure for the generation of release volumes.
1. Clustering of release volumes.

**Operational submodule**
The operational submodule assigns probabilities to the release volumes based on a shakemap. This encompass:

1. Calculation of cumulative distribution PGA (ensemble).
1. Calculation of exceedance probabilities of the earthquake-induced slope displacements.
1. Calculation of release probabilities.


## Input
Example input files can be downloaded from... These include
 - Preparataional submodule: Bathymetry and soilparameters. The soilparameters are given as a JSON describing the parameters associated with a raster region file. 
 - The ensemble of shakemaps given in JSON format and an associated csv file with source parameters.


## Execution
Clone the repository and use poetry to set up an environment. Execution of the preparational and the operational submodules can be carried out either using the scripts (see `Makefile`) or through respective jupyter notebooks.
Configurations and parameters associoated with each step are specified directly in the scripts `src/preparational.py` and `src/operational.py` or in the notebooks. The operational and the preparational steps may be excecuted from the Makefile. An empty file named `completed` in a subfolder is used to signify that the step has been carried out. Note that the `volumes` folder contains the `volumes.db` file, a file that will be updated when running the operational step. To clean up the operational step there is a script `clean_operational.py`.

After execution of both submodules, the generated `rundir` has the following folder structure:
```
├── aggregation
├── displacements
├── shakemaps
├── slope_analysis
│   ├── fos
│   │   ├── cummulative
│   │   └── quantiles
│   └── yield_acceleration
│       ├── cummulative
│       └── quantiles
├── triangulation
└── volumes
```

## Dependencies
 - Python dependencies are specified in the `pyproject.toml` file.

 - [Whitebox tools](https://github.com/jblindsay/whitebox-tools) is applied for the extraction of basic topographic features like slope and aspect. Alternatively, slope and aspect may be given as input.

 - [GDAL](https://gdal.org/en/stable/) GDAL is applied for some basic raster operations in the preparational part of the module.