
# Makefile with some basic execution commands.

# Default input, for other paths provide input when running the commands:
# make volumes rootdir=c:/ROOTDIR
ROOTDIR ?= /home/ebr/projects/release-volume-sampler
REGION ?= messina_20250806
RUNDIR ?= $(ROOTDIR)/generated/$(REGION)# Defined path in preparational.py


# Multistep proceedures - Parameters set in python scripts
# nohup poetry run python src/preparational.py --ROOTDIR $(ROOTDIR) --RUNDIR $(RUNDIR) > output_vol.log 2>&1 &
# nohup poetry run python src/operational.py --ROOTDIR $(ROOTDIR) --RUNDIR $(RUNDIR) > output_prob.log 2>&1 &
volumes:
	@echo " Execute slope analysis and sample volumes..."
	poetry run python src/preparational.py --rootdir $(ROOTDIR) --region $(REGION)

probabilities:
	@echo " Assign probabilities to volumes..."
	poetry run python src/operational.py --rootdir $(ROOTDIR) --rundir $(RUNDIR)

plots:
	@echo " Plot $(TYPE).."
	@for folder in $(shell find $(RUNDIR) -type d -name "$(TYPE)"); do \
		echo "Plot content $$folder"; \
		poetry run python src/plot.py "$$folder"; \
	done

clean-folder:
	@if [ -z "$(FOLDER)" ]; then \
		echo "Usage: make clean-folder FOLDER=subfolder_name"; \
		exit 1; \
	fi; \
	echo "Deleting $(RUNDIR)/$(FOLDER)"; \
	rm -rf "$(RUNDIR)/$(FOLDER)"

clean:
	@echo " Delete all output in $(RUNDIR)..."
	rm -rf $(RUNDIR)

# Help target
help:
	@echo "Makefile for Generation of Release Volumes"
	@echo "Usage:"
	@echo "  make volumes - Run preparational script - Creates volume database."
	@echo "  make probabilities - Run operational script - Assigns probabilities to volumes."
	@echo "  make plots TYPE=shakemaps - Plot a specific type (shakemaps, cumulative, quantiles, displacements)."
	@echo "  make clean - Empty generated folder."
	@echo "  make clean-folder - Empty subfolder of the rundir. Usage: make clean-folder FOLDER=subfolder_name"
	@echo "  make help - Display this help message."

.PHONY: clean help volumes probabilities clean-folder plots
