
# Makefile with some basic execution commands.

# Note that the main workflow is implemented in main.py. 
# This Makefile is just for convenience during development and testing.

RUNDIR ?= generated/messina_001_20241008_115914
GENERATED_DIR = generated

run:
	@echo " Run workflow from main.py..."
	poetry run python src/main.py

clean:
	@echo " Delete all output in $(GENERATED_DIR)..."
	rm -r $(GENERATED_DIR)/*

# Help target
help:
	@echo "Makefile for Generation of Release Volumes"
	@echo "Usage:"
	@echo "  make run - Run main.py"
	@echo "  make clean   - Remove the gener"
	@echo "  make help    - Display this help message"

.PHONY: run clean help
