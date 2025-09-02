import os
import shutil
import argparse

"""
Script used for creation of minimal rundir structure to support the operational module.
"""

def copytree_exclude(src, dst, exclude_paths):
    """
    Copy src to dst, excluding any files or directories whose relative path from src is in exclude_paths.
    """
    exclude_set = set(os.path.normpath(p) for p in exclude_paths)
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        # Skip excluded directories
        for d in list(dirs):
            rel_dir = os.path.normpath(os.path.join(rel_root, d))
            if rel_dir in exclude_set:
                dirs.remove(d)
                print(f"Excluding directory: {os.path.join(root, d)}")
        # Copy files
        for f in files:
            rel_file = os.path.normpath(os.path.join(rel_root, f))
            if rel_file in exclude_set:
                print(f"Excluding file: {os.path.join(root, f)}")
                continue
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst, rel_root, f)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", required=True, help="Path to rundir")
    args = parser.parse_args()

    # List the relative paths (from rundir) you want to remove
    paths_to_remove = [
        "slope_analysis/fos",
        "slope_analysis/yield_acceleration/quantiles",
        "slope_analysis/slopeanalysis.log",
        "slope_analysis/completed",
        "triangulation/triangulation.png",
        "triangulation/triangulation.log",
        "triangulation/triangulate.log",
        "triangulation/completed",
        "volumes/rasters",
        "volumes/volume_representatives.csv",
        "volumes/cluster_analysis",
        "volumes/release_characteristics-p_fos_seed.png",
        "volumes/release_distribution-p_fos_seed.png",
        "volumes/volume_sampler.log",
        "volumes/db_handler.log",
        "volumes/completed",
        "volumes/cluster.log",
        "aspect.tif",
        "bathy.tif",
        "settings.json",
        "slope.tif",
        "slumap.tif",
        "soilparams.json",
        "soilregions.tif",
        "preparational.log",
        "preparational_external_software.txt",
        "completed",
        "operational.log"
        # Add more as needed
    ]
    parent_dir = os.path.dirname(args.rundir.rstrip(os.sep))
    rundir_basename = os.path.basename(args.rundir.rstrip(os.sep))
    rundir_minimal = os.path.join(parent_dir, rundir_basename + "_minimal")
    if os.path.exists(rundir_minimal):
        print(f"Removing existing minimal rundir: {rundir_minimal}")
        shutil.rmtree(rundir_minimal)
    print(f"Creating minimal rundir: {rundir_minimal}")
    os.makedirs(rundir_minimal, exist_ok=True)

    copytree_exclude(args.rundir, rundir_minimal, paths_to_remove)
    print("Minimal rundir created.")

