# Makefile for Singularity container management


# Grass variables
MAIN = /home/sfr/release-volume-sampler
IMAGE_DIR ?= $(MAIN)/images
IMAGE_NAME = grass.sif
DEF_FILE = grass.def
IMAGE_PATH = $(IMAGE_DIR)/$(IMAGE_NAME)
RUNDIR ?= $(MAIN)/generated/

# Slopeunit variables
thresh = 4000000
areamin = 1000000
cvmin = 0.2
rf = 20
bathy_file = $(MAIN)/input/bathy/localMessinaBathy.tif
slopeunits_path = $(MAIN)/slopeunits


# Convert bathymetri to utm coordinates
bathy_convert:
	poetry run python convert_bathy.py --bathy_file $(bathy_file)

# Build the Singularity container
build:
	@echo "Building Singularity container..."
	singularity build --fakeroot $(IMAGE_PATH) $(DEF_FILE)

# Run a shell inside the Singularity container
shell:
	@echo "Running a shell in the Singularity container..."
	singularity shell $(IMAGE_PATH)

start_project:
	cd $(RUNDIR) && grass grassdata/PERMANENT

#Create slopeunit file
single:
	poetry run python run_single.py --thresh $(thresh) --areamin $(areamin) --cvmin $(cvmin) --rf $(rf) --IMAGE_PATH $(IMAGE_PATH) --bathy_file $(bathy_file)

# Create many slopeuntis based on a larget set of input variables
array:
	poetry run python run.py --IMAGE_PATH $(IMAGE_PATH) --bathy_file $(bathy_file)

# Plote the slopeunits
plot:
	poetry run python plot_slopeunits.py --bathy_file $(bathy_file) --run_path $(slopeunits_path)

# Clean up the built image
delete_image:
	@echo "Delete singularity image $(IMAGE_PATH)..."
	rm -f $(IMAGE_PATH) 
 
clean:
	@echo " Delete grassdata and slumap.tif from $(RUNDIR)..."
	rm -r $(RUNDIR)/grassdata
	rm $(RUNDIR)/slumap.tif
	rm $(RUNDIR)/slumap_clean.tif

# Help target
help:
	@echo "Makefile for Singularity container management"
	@echo "Usage:"
	@echo "  make build   - Build the Singularity container"
	@echo "  make shell   - Run a shell inside the Singularity container"
	@echo "  make clean   - Remove the generated output of the run"
	@echo "  make help    - Display this help message"

.PHONY: all build shell clean help
