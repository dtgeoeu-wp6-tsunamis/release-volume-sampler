
# Makefile with some basic execution commands.

RUNDIR = /home/ebr/projects/release-volume-sampler/generated/messina_001
PROJECTED_BATHY = /home/ebr/projects/release-volume-sampler/input/bathy/messina_001/bathy_truncated_pr.tif
GENERATED_DIR = generated

IMAGE_DIR = /home/ebr/projects/release-volume-sampler/images
IMAGE_NAME = grass.sif
DEF_FILE = grass.def
IMAGE_PATH = $(IMAGE_DIR)/$(IMAGE_NAME)
SLOPEUNIT_FILE = $(GENERATED_DIR)/slopeunits/slumap.tif


slopeunits: clean-slopeunits
	@echo " Calculating slopeunits..."
	poetry run python src/slopeunits/slopeunits.py $(GENERATED_DIR) $(IMAGE_PATH) $(PROJECTED_BATHY)

# Multistep proceedures - Parameters set in python scripts
volumes: clean-volumes
	@echo " Execute slope analysis and sample volumes..."
	poetry run python -m src.regional_analysis

probabilities:
	@echo " Assign probabilities to volumes..."
	poetry run python -m src.run

plots:
	@echo " Plot cummulatives.."
	@for folder in $(shell find $(RUNDIR) -type d -name "cummulative"); do \
		echo "Plot content $$folder"; \
		poetry run python -m src.plot "$$folder"; \
	done
	
	@echo " Plot quantiles.."
	@for folder in $(shell find $(RUNDIR) -type d -name "quantiles"); do \
		echo "Plot content $$folder"; \
		poetry run python -m src.plot "$$folder"; \
	done
	
	@echo " Plot displacements.."
	@for folder in $(shell find $(RUNDIR) -type d -name "displacements"); do \
		echo "Plot content $$folder"; \
		poetry run python -m src.plot "$$folder"; \
	done
	@echo " Plot shakemap.."
	@for folder in $(shell find $(RUNDIR) -type d -name "shakemaps"); do \
		echo "Plot content $$folder"; \
		poetry run python -m src.plot "$$folder"; \
	done

clean-slopeunits:
	@echo " Delete all output in $(GENERATED_DIR)/slopeunits..."
	rm -rf $(GENERATED_DIR)/slopeunits*

clean-volumes:
	@echo " Delete all output in $(RUNDIR)..."
	rm -rf $(RUNDIR)

clean: clean-slopeunits clean-volumes

# Help target
help:
	@echo "Makefile for Generation of Release Volumes"
	@echo "Usage:"
	@echo "  make run - Run run.py"
	@echo "  make analysis - Run regional_analysis.py"
	@echo "  make clean - Empty generated folder"
	@echo "  make plots - Plot output."
	@echo "  make help - Display this help message"

.PHONY: clean help clean-slopeunits clean-analysis
