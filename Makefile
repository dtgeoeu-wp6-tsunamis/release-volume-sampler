
# Makefile with some basic execution commands.

# Note that the main workflow is implemented in main.py. 
# This Makefile is just for convenience during development and testing.
# Pass RUNDIR as argument:
# make plots RUNDIR=...

RUNDIR ?= generated/messina_001_20241008_115914
GENERATED_DIR = generated

run:
	@echo " Run workflow from main.py..."
	poetry run python src/main.py

clean:
	@echo " Delete all output in $(GENERATED_DIR)..."
	rm -r $(GENERATED_DIR)/*

plots:
	@echo " Plot yield acceleration..."
	python src/plot.py $(RUNDIR)/yield_acceleration/cummulative --logscale	
	python src/plot.py $(RUNDIR)/yield_acceleration/quantiles --logscale	
	
	@echo " Plot fos..."
	python src/plot.py $(RUNDIR)/fos/cummulative --logscale	
	python src/plot.py $(RUNDIR)/fos/quantiles --logscale	
	
	@echo " Plot shakemaps..."
	python src/plot.py $(RUNDIR)/shakemaps --logscale	
	python src/plot.py $(RUNDIR)/shakemaps --logscale	

# Help target
help:
	@echo "Makefile for Generation of Release Volumes"
	@echo "Usage:"
	@echo "  make run - Run main.py"
	@echo "  make clean - Empty generated folder"
	@echo " make plots - Plot yield acceleration, fos and shakemaps"
	@echo "  make help    - Display this help message"

.PHONY: run clean help
