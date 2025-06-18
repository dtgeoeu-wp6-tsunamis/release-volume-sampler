import subprocess
import os


# Define the Singularity image path and the command to run inside the container
singularity_image = "/mnt/c/Users/SFr/Work/VolumeSampler/images/grass.sif"  # Replace with your Singularity image file path
command_load_bathy = ["grass grassdata/PERMANENT --exec r.in.gdal input=bathy_projected_truncated.tif output=bathy --overwrite"]

command_calc_slopes = ["grass grassdata/PERMANENT --exec r.slope.aspect elevation=bathy slope=slope aspect=aspect format=degrees --overwrite"]
command_save_slope1 = ["grass grassdata/PERMANENT --exec r.out.gdal input=slope output=slope.tif --overwrite"]
command_save_slope2 = ["grass grassdata/PERMANENT --exec r.out.gdal input=aspect output=aspect.tif --overwrite"]

for thresh in range(500000,5000000,500000):
    for areamin in range(500000,5000000,500000):
        for cvmin in range(1,10,1):
            cvmin = cvmin/10
            for rf in range(10,100,10):
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



print('Hei!')
#subprocess.run('grass grassdata/PERMANENT --exec r.in.gdal input=bathy_projected_truncated.tif output=bathy --overwrite')
#subprocess.run('bash -c grass'+' '+r'/mnt/c/Users/SFr/Work/VolumeSampler/slopeunits/grassdata/PERMANENT')