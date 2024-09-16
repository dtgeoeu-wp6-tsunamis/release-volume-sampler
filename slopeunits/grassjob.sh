#!/bin/bash
bathyfile=$1

# Load bathy
r.in.gdal input=$bathyfile output=bathy

# Calculate slopes
r.slope.aspect elevation=bathy slope=slope aspect=aspect format=degrees
r.out.gdal input=slope output=slope.tif
r.out.gdal input=aspect output=aspect.tif

# Run slopeunits
#r.slopeunits demmap=bathy slumap=slumap thresh=5000000 areamin=100000 cvmin=0.35 rf=10 maxiteration=10 cleansize=20000 slumapclean=slumap_clean
#r.out.gdal input=slumap output=slumap.tif
#r.out.gdal input=slumap_clean output=slumap_clean.tif
