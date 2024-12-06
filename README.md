# Earthquake Induced Recursive Release Volume Sampler
A tool designed to sample potential submarine release volumes triggered by earthquakes.



## Dependencies
 - [Whitebox tools](https://github.com/jblindsay/whitebox-tools) is applied for the extraction of basic topographic features like slope and aspect.


## Extraction of slopeunits
We apply `r.slopeunits` for delineation of the bathymetry into terrain uints ([Alvioli et al, 2016](https://doi.org/10.5194/gmd-9-3975-2016), [Alvioli et al, 2020](https://doi.org/10.1016/j.geomorph.2020.107124)). The script is written using [GRASS GIS](https://grass.osgeo.org/), and is currently set up to be executed in a singularity container. To apply the container set `IMAGE_DIR` and `RUNDIR` in  `slopeunits/Makefile` (or as environment variables) and run `make build` and `make run` to build and run the singularity image respectively. To change the parameters edit `slopeunits/grassjob.sh`.