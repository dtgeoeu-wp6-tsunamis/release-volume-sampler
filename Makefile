
# Makefile with some basic execution commands.

# Default input, for other paths provide input when running the commands:
# make volumes RUNDIR=c:/rundir
RUNDIR ?= /home/sfr/release-volume-sampler
RESDIR ?= $(RUNDIR)/generated/messina_001

# Multistep proceedures - Parameters set in python scripts
# nohup poetry run python src/preparational.py --rundir $(RUNDIR) --resdir $(RESDIR) > output_vol.log 2>&1 &
# nohup poetry run python src/operational.py --rundir $(RUNDIR) --resdir $(RESDIR) > output_prob.log 2>&1 &
volumes:
	@echo " Execute slope analysis and sample volumes..."
	poetry run python src/preparational.py --rundir $(RUNDIR) --resdir $(RESDIR)

probabilities:
	@echo " Assign probabilities to volumes..."
	poetry run python src/operational.py --rundir $(RUNDIR) --resdir $(RESDIR)

# Plot output make plots RUNDIR=...
plots:
	@echo " Plot cumulatives.."
	@for folder in $(shell find $(RESDIR) -type d -name "cumulative"); do \
		echo "Plot content $$folder"; \
		poetry run python -m src.plot "$$folder"; \
	done
	
	@echo " Plot quantiles.."
	@for folder in $(shell find $(RESDIR) -type d -name "quantiles"); do \
		echo "Plot content $$folder"; \
		poetry run python -m src.plot "$$folder"; \
	done
	
	@echo " Plot displacements.."
	@for folder in $(shell find $(RESDIR) -type d -name "displacements"); do \
		echo "Plot content $$folder"; \
		poetry run python -m src.plot "$$folder"; \
	done

	@echo " Plot shakemap.."
	@for folder in $(shell find $(RESDIR) -type d -name "shakemaps"); do \
		echo "Plot content $$folder"; \
		poetry run python -m src.plot "$$folder"; \
	done


clean:
	@echo " Delete all output in $(RESDIR)..."
	rm -rf $(RESDIR)

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
