
# Makefile with some basic execution commands.
RUNDIR ?= /home/ebr/projects/release-volume-sampler/generated/messina_001

# Multistep proceedures - Parameters set in python scripts
volumes:
	@echo " Execute slope analysis and sample volumes..."
	poetry run python src/preparational.py

probabilities:
	@echo " Assign probabilities to volumes..."
	poetry run python src/operational.py

# Plot output make plots RUNDIR=...
plots:
	@echo " Plot cumulatives.."
	@for folder in $(shell find $(RUNDIR) -type d -name "cumulative"); do \
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


clean:
	@echo " Delete all output in $(RUNDIR)..."
	rm -rf $(RUNDIR)

# Help target
help:
	@echo "Makefile for Generation of Release Volumes"
	@echo "Usage:"
	@echo "  make volumes - Run preparational script - Creates volume database."
	@echo "  make probabilities - Run operational script - Assigns probabilities to volumes."
	@echo "  make clean - Empty generated folder."
	@echo "  make plots - Plot rasters."
	@echo "  make help - Display this help message."

.PHONY: clean help volumes probabilities
