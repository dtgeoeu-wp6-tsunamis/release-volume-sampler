# Makefile for Singularity container management

# Variables
IMAGE_DIR = /home/ebr/projects/release-volume-sampler/images
IMAGE_NAME = grass.sif
DEF_FILE = grass.def
IMAGE_PATH = $(IMAGE_DIR)/$(IMAGE_NAME)
RUNDIR ?= /home/ebr/projects/release-volume-sampler/generated/messina_001_20240819_132742

# Default target
all: build

# Build the Singularity container
build:
	@echo "Building Singularity container..."
	singularity build --fakeroot $(IMAGE_PATH) $(DEF_FILE)

# Run a shell inside the Singularity container
shell:
	@echo "Running a shell in the Singularity container..."
	singularity shell $(IMAGE_PATH)

# Run container
run:
	@echo "Running the Singularity container..."
	singularity run $(IMAGE_PATH) $(RUNDIR)

# Clean up the built image
clean:
	@echo "Delete singularity image $(IMAGE_PATH)..."
	rm -f $(IMAGE_PATH)
	
	@echo " Delete grassdata and slumap.tif from $(RUNDIR)..."
	rm -r $(RUNDIR)/grassdata
	rm $(RUNDIR)/slumap.tif

# Help target
help:
	@echo "Makefile for Singularity container management"
	@echo "Usage:"
	@echo "  make         - Build the Singularity container (default target)"
	@echo "  make build   - Build the Singularity container"
	@echo "  make run     - Run the container for the given rundir."
	@echo "  make shell   - Run a shell inside the Singularity container"
	@echo "  make clean   - Remove the built Singularity image"
	@echo "  make help    - Display this help message"

.PHONY: all build shell clean help
