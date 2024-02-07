import argparse
from microbundlepillartrack import pillar_analysis as pa
from microbundlepillartrack import create_pillar_mask as cpm
from pathlib import Path
import numpy as np
import glob
import os
import traceback


failed_paths = []

paths = glob.glob('/Users/hibakobeissi/Desktop/Pillar_Tracking/Test_Examples/Sam_Data/fibroTUG_033/test_reset/*')
paths = sorted(paths)

for ii in range(len(paths)):
    try: 
        input_folder_str = paths[ii]
        if os.path.isdir(input_folder_str):
            print('folder',input_folder_str)
            self_path_file = Path(__file__)
            self_path = self_path_file.resolve().parent
            input_folder = self_path.joinpath(input_folder_str).resolve()

            '''Movie parameters: frames per second(fps) and length scale (ls) as micrometers/pixel'''
            '''Type1 movie parameters'''
            # fps = 30 
            # ls = 4

            """Type2 movie parameters"""        
            fps = 65 
            ls = 1.1013
            
            '''Indicate microbundle type'''   
            microbundle_type = "type2" # Microbundle type can be either "type1" or "type2"

            '''Indicate checkpoint path'''
            checkpoint_path = Path('/Users/hibakobeissi/Desktop/Pillar_Tracking/MicroBundlePillarTrack_Recent_src/src/microbundlepillartrack')

            '''Pillar stiffness can be directly provided: replace `None` by a value''' 
            '''Type 1 pillar stiffness'''
            #pillar_stiffnes = 2.677 # Provide this value in microNewton per micrometer (uN/um) 
            '''Type 2 pillar stiffness: soft pillars'''
            pillar_stiffnes = 0.41 # Provide this value in microNewton per micrometer (uN/um) 
            '''Type 2 pillar stiffness: stiff pillars'''
            #pillar_stiffnes = 1.2 # Provide this value in microNewton per micrometer (uN/um)
            # pillar_stiffnes = None # If pillar stiffness is to be calculated based on the below inputs       
            ''' Or calculated based on the specified pillar parameters (Change as suitable)'''
            pillar_profile = 'circular' # Pillar profile can be either "rectangular" or "circular"
            pdms_E = 1.61 # Provide this value in MPa
            # If rectangular: 
            pillar_width = 163 # Provide this value in micrometer (um)
            pillar_thickness = 33.2 # Provide this value in micrometer (um)
            # If circular:
            pillar_diameter = 400 # Provide this value in micrometer (um)

            pillar_length = 199.3 # Provide this value in micrometer (um)
            force_location = 163 # Provide this value in micrometer (um)

            pillar_orientation_vector = None
            """Type1 tissue depth"""        
            # tissue_depth = 350 # Provide this value in micrometer (um) (type 1)
            """Type2 tissue depth"""  
            tissue_depth = 10 # Provide this value in micrometer (um) (type 2)
            
            '''generate pillar masks'''
            cpm.run_create_pillar_mask(input_folder, checkpoint_path, microbundle_type)
            
            ''' Set to `True` to eliminate drift if observed in pillar results'''
            split = False
            '''run and visualize pillar tracking'''
            pa.run_pillar_tracking(input_folder, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pdms_E, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, ls, fps, split)
            pa.visualize_pillar_tracking(input_folder, ls, fps, split)

    except IndexError:
        print(traceback.format_exc())
        with open(str(input_folder.resolve().parent)+'/Errors.txt', "a") as f:
            traceback.print_exc(file=f)
        failed_paths.append(str(input_folder))
        pass
    except OSError:
        print(traceback.format_exc())
        with open(str(input_folder.resolve().parent)+'/Errors.txt', "a") as f:
            traceback.print_exc(file=f)
        failed_paths.append(str(input_folder))
        pass
    except AttributeError:
        print(traceback.format_exc())
        with open(str(input_folder.resolve().parent)+'/Errors.txt', "a") as f:
            traceback.print_exc(file=f)
        failed_paths.append(str(input_folder))
        pass

if len(failed_paths) > 0:
    np.savetxt(str(input_folder.resolve().parent)+'/failed_paths.txt',failed_paths, fmt="%s")   
