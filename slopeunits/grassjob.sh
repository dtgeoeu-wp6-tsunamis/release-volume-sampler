#!/bin/bash
bathyfile=$1

# Load bathy
r.in.gdal input=$bathyfile output=bathy --overwrite

# Calculate slopes
r.slope.aspect elevation=bathy slope=slope aspect=aspect format=degrees --overwrite
r.out.gdal input=slope output=slope.tif --overwrite
r.out.gdal input=aspect output=aspect.tif --overwrite

# Run slopeunits
r.slopeunits demmap=bathy slumap=slumap thresh=1000000 areamin=500000 cvmin=0.6 rf=30 maxiteration=20 cleansize=10000 slumapclean=slumap_clean --overwrite
r.out.gdal input=slumap output=slumap.tif --overwrite
r.out.gdal input=slumap_clean output=slumap_clean.tif --overwrite
