./grassjob.sh bathy_projected_truncated.tif
r.in.gdal input=bathy_projected_truncated.tif output=bathy --overwrite
./grassjob.sh bathy_projected_truncated.tif
r.slopeunits demmap=bathy slumap=slumap thresh=1000000 areamin=500000 cvmin=0.6 rf=30 maxiteration=20 cleansize=10000 slumapclean=slumap_clean --overwrite
r.slopeunits demmap=bathy slumap=slumap thresh=1000000 areamin=1000000 cvmin=0.6 rf=30 maxiteration=20 cleansize=10000 slumapclean=slumap_clean --overwrite
r.slopeunits demmap=bathy slumap=slumap thresh=1000000 areamin=1000000 cvmin=0.6 rf=30 maxiteration=20 cleansize=50000 slumapclean=slumap_clean --overwrite
r.out.gdal input=slumap output=slumap.tif --overwrite
r.out.gdal input=slumap_clean output=slumap_clean.tif --overwrite
r.slopeunits demmap=bathy slumap=slumap thresh=1000000 areamin=1000000 cvmin=0.8 rf=30 maxiteration=40 cleansize=50000 slumapclean=slumap_clean --overwrite
r.out.gdal input=slumap output=slumap.tif --overwrite
r.out.gdal input=slumap_clean output=slumap_clean.tif --overwrite
r.slopeunits demmap=bathy slumap=slumap thresh=1000000 areamin=1000000 cvmin=0.2 rf=30 maxiteration=40 cleansize=50000 slumapclean=slumap_clean --overwrite
exit
exit
exit
exit
exit
exit
exit
r.slopeunits demmap=bathy slumap=slumap thresh=1000000 areamin=500000 cvmin=0.6 rf=30 maxiteration=2 cleansize=10000 slumapclean=slumap_clean --overwrite
exit
