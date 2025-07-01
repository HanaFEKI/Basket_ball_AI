.PHONY: all run_terrain run_tracker

all: run_terrain run_tracker

run_terrain:
	$(MAKE) -C ../Terrain_Detection

run_tracker:
	@echo "Running tracker "
	python3 run_adaptive_homography.py
