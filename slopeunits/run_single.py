import subprocess
import os
import argparse

# Rundir tas inn som input
parser = argparse.ArgumentParser(description="Release volume sampler")
parser.add_argument('--thresh', required=True, help='Threshold parameter')
parser.add_argument('--areamin', required=True, help='areamin')
parser.add_argument('--cvmin', required=True, help='cvmin')
parser.add_argument('--rf', required=True, help='rf')
parser.add_argument('--IMAGE_PATH', required=True, help='Path to the singularity image')
parser.add_argument('--bathy_file', required=True, help='Bathymetri')
args = parser.parse_args()
thresh = args.thresh
areamin = args.areamin
cvmin = args.cvmin
rf = args.rf
singularity_image = args.IMAGE_PATH
bathy_file = args.bathy_file

#bathy_file = "bathy_projected_truncated.tif"
#bathy_file = "test.tif"
bathy_file = bathy_file[:-4]+"_projected_truncated.tif"

#singularity_image = "/home/sfr/release-volume-sampler/images/grass.sif"  # Replace with your Singularity image file path
#thresh = 4000000
#areamin = 1000000
#cvmin = 0.2
#rf = 20

# Define the Singularity image path and the command to run inside the container
command_load_bathy = ["grass grassdata/PERMANENT --exec r.in.gdal input="+bathy_file+" output=bathy --overwrite"]
command_calc_slopes = ["grass grassdata/PERMANENT --exec r.slope.aspect elevation=bathy slope=slope aspect=aspect format=degrees --overwrite"]
command_save_slope1 = ["grass grassdata/PERMANENT --exec r.out.gdal input=slope output=slope.tif --overwrite"]
command_save_slope2 = ["grass grassdata/PERMANENT --exec r.out.gdal input=aspect output=aspect.tif --overwrite"]



mdir = 't' + str(thresh) + '_a' + str(areamin)+'_c'+str(cvmin)+'_rf'+str(rf)

print(mdir + ' Starting')
outfold = os.path.join('Results',mdir)
os.makedirs(outfold,exist_ok=True)

#command_calc_slopesU1 = ["grass grassdata/PERMANENT --exec r.slopeunits demmap=bathy slumap=slumap thresh="+str(thresh)+" areamin="+str(areamin)+" cvmin="+str(cvmin)" rf="+str(rf)]
command_calc_slopesU = ["grass grassdata/PERMANENT --exec r.slopeunits demmap=bathy slumap=slumap thresh="+str(thresh)+" areamin="+str(areamin)+" cvmin="+str(cvmin)+" rf="+str(rf)+" maxiteration=20 cleansize=10000 slumapclean=slumap_clean --overwrite"]
command_save_slopeU1 = ["grass grassdata/PERMANENT --exec r.out.gdal input=slumap output="+os.path.join(outfold,"slumap.tif")+" --overwrite"]
command_save_slopeU2 = ["grass grassdata/PERMANENT --exec r.out.gdal input=slumap_clean output="+os.path.join(outfold,"slumap_clean.tif")+" --overwrite"]




# Construct the full singularity exec command
singularity_command1 = ["singularity", "exec", singularity_image, "bash", "-c"] + [" ".join(command_load_bathy)]
singularity_command2 = ["singularity", "exec", singularity_image, "bash", "-c"] + [" ".join(command_calc_slopes)]
singularity_command3 = ["singularity", "exec", singularity_image, "bash", "-c"] + [" ".join(command_save_slope1)]
singularity_command4 = ["singularity", "exec", singularity_image, "bash", "-c"] + [" ".join(command_save_slope1)]
singularity_command5 = ["singularity", "exec", singularity_image, "bash", "-c"] + [" ".join(command_calc_slopesU)]
singularity_command6 = ["singularity", "exec", singularity_image, "bash", "-c"] + [" ".join(command_save_slopeU1)]
singularity_command7 = ["singularity", "exec", singularity_image, "bash", "-c"] + [" ".join(command_save_slopeU2)]

command_str = " ".join(singularity_command1)
print(command_str)
result1 = subprocess.run(singularity_command1, capture_output=True, text=True, check=True)
#print(result1)
#result2 = subprocess.run(singularity_command2, capture_output=True, text=True, check=True)
#print(result2)
#result3 = subprocess.run(singularity_command3, capture_output=True, text=True, check=True)
#print(result3)
#result4 = subprocess.run(singularity_command4, capture_output=True, text=True, check=True)
#print(result4)
result5 = subprocess.run(singularity_command5, capture_output=True, text=True, check=True)
#print(result5)
result6 = subprocess.run(singularity_command6, capture_output=True, text=True, check=True)
#print(result6)
result7 = subprocess.run(singularity_command7, capture_output=True, text=True, check=True)
#print(result7)

print(mdir + ' Finished')

#subprocess.run('grass grassdata/PERMANENT --exec r.in.gdal input=bathy_projected_truncated.tif output=bathy --overwrite')
#subprocess.run('bash -c grass'+' '+r'/mnt/c/Users/SFr/Work/VolumeSampler/slopeunits/grassdata/PERMANENT')