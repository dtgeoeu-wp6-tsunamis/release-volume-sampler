# Original README:
for Windows users: this software was designed and tested for use under Linux.
If you downloaded the package with a Windows e-mail client, it might be
necessary to run "dos2unix" on all files contained in the package, i.e. run:
```bash
dos2unix r.slopeunits
dos2unix clean_method_3.sh
```

Please make sure that both files are executable before attempting to use them.
If not, grant execution privileges to both files, i.e. run:

```bash
chmod +x r.slopeunits
chmod +x clean_method_3.sh
```

# Preprocessing of DEM. 
Set non-negative bathymetry values to nodata.
```bash
gdal_calc.py -A bathy/localMessinaBathy.tif --outfile=truncated.tif --calc="A*(A<0)" --NoDataValue=0
```

Convert from degrees to UTM (https://epsg.io/6709) 
```
gdalwarp -t_srs EPSG:6709 /home/ebr/projects/release-volume-sampler/bathy/messina_001/bathy_truncated.tif bathy.tif
```

# Run Instructions
https://geomorphology.irpi.cnr.it/tools/slope-units:

The software have been tested starting on Ubuntu 14.04 LTS server but        
should run on a generic GNU/Linux machine.                
	
	Requirements: our software require GRASS GIS 7.*, Python, Bash.                
	
	The two files contained in the tar.gz archives (one python script, one bash script) must be copied        
	in the “scripts” folder of the GRASS GIS installation directory. They must be executable (see below).        
	A typical approach, for the installation, is (replace grass78 for previous versions):               

	# open a bash shell               

	# change working directory to the download folder, e.g.:                

	cd /home/$USER/Downloads                

	# move the files into the scripts folder of the grass installation, e.g.:         

	mv r.slopeunits /usr/lib/grass78/scripts          

	mv clean_method_3.sh /usr/lib/grass78/scripts               

	# give execution rights                

	chmod ugo+x /usr/lib/grass78/scripts/r.slopeunits               

	chmod ugo+x /usr/lib/grass78/scripts/clean_method_3.sh                

	# run GRASS GIS in the location containing the digital elevation model type and:               

	r.slopeunits --help                

	# to see the options. The minimal command line to obtain a slope units delineation is                

	r.slopeunits demmap=[dem] slumap=[output_SU_map] thresh=[t, square meters] circularvariance=[c] areamin=[a, square meters] reductionfactor=[r, r>2] maxiteration=[max number of iterations]

# Build container
https://docs.sylabs.io/guides/3.0/user-guide/quick_start.html

Build image:
```bash
singularity build --fakeroot images/grass.sif grass.def
```

Test:
```bash
singularity shell images/grass.sif
```
https://grass.osgeo.org/grass78/manuals/helptext.html
https://grass.osgeo.org/grass84/manuals/index.html

 grass --tmp-project bathy.tif

 ### Create grass project from bathy file
 grass -e -c bathy.tif /home/ebr/grassdata/test
 ### Import bathy
 grass /home/ebr/grassdata/test/PERMANENT --exec r.external input=bathy.tif output=bathy
 ### Run slopeunits.
 r.slopeunits demmap=bathy slumap=slumap thresh=100000 areamin=10000 cvmin=10 rf=3 maxiteration=3
 ### Export to file
 r.out.gdal input=slumap output=slumap.tif