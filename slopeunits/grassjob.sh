#!/bin/bash
rundir=$1

# Load bathy
r.in.gdal input=$rundir/bathy.tif output=bathy

# Run slopeunits
r.slopeunits demmap=bathy slumap=slumap thresh=100000 areamin=10000 cvmin=10 rf=3 maxiteration=3

# Export
r.out.gdal input=slumap output=$rundir/slumap.tif