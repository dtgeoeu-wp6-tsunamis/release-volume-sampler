import os
import shutil
import argparse
from rvsampler.database_handler import VolumeDatabaseHandler
from stat import S_IWRITE

def remove_readonly(func, path, excinfo):
    os.chmod(path, S_IWRITE)
    func(path)

def cleanup_operational(rundir):
    folders = ["shakemaps", "displacements", "aggregation"]
    for folder in folders:
        path = os.path.join(rundir, folder)
        if os.path.exists(path):
            print(f"Removing folder: {path}")
            shutil.rmtree(path, onerror=remove_readonly)
        else:
            print(f"Folder not found (skipped): {path}")

    with VolumeDatabaseHandler(rundir) as db:
        print("Removing all p_shake columns from the database...")
        db.drop_p_shake_columns()
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", required=True)
    args = parser.parse_args()
    cleanup_operational(args.rundir)