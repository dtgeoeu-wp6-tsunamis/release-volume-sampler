
# Makefile with some basic execution commands.

RUNDIR = /home/ebr/projects/release-volume-sampler/generated/messina_001
PROJECTED_BATHY = /home/ebr/projects/release-volume-sampler/input/bathy/messina_001/bathy_truncated_pr.tif
GENERATED_DIR = generated

IMAGE_DIR = /home/ebr/projects/release-volume-sampler/images
IMAGE_NAME = grass.sif
DEF_FILE = grass.def
IMAGE_PATH = $(IMAGE_DIR)/$(IMAGE_NAME)
SLOPEUNIT_FILE = $(GENERATED_DIR)/slopeunits/slumap_clean.tif


slopeunits: clean-slopeunits
	@echo " Calculating slopeunits..."
	poetry run python src/slopeunits/slopeunits.py $(GENERATED_DIR) $(IMAGE_PATH) $(PROJECTED_BATHY)

analysis: $(SLOPEUNIT_FILE) 
	@echo " Run regional slope analysis and extract preselection of volumes..."
	poetry run python src/regional_analysis.py

run: analysis
	@echo " Run workflow from main.py..."
	poetry run python src/run.py

plots:
	@echo " Plot cummulatives.."
	@for folder in $(shell find $(RUNDIR) -type d -name "cummulative"); do \
		echo "Plot content $$folder"; \
		poetry run python src/plot.py "$$folder" --logscale; \
	done
	
	@echo " Plot quantiles.."
	@for folder in $(shell find $(RUNDIR) -type d -name "quantiles"); do \
		echo "Plot content $$folder"; \
		poetry run python src/plot.py "$$folder" --logscale; \
	done

clean-slopeunits:
	@echo " Delete all output in $(GENERATED_DIR)/slopeunits..."
	rm -rf $(GENERATED_DIR)/slopeunits*

clean-analysis:
	@echo " Delete all output in $(RUNDIR)..."
	rm -rf $(RUNDIR)

clean: clean-slopeunits clean-analysis

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
