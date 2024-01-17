from microbundlepillartrack import image_analysis as ia
from pathlib import Path
from typing import List
import imageio
import glob
import os 


def tif_to_TIFF_frames(folder_path: Path) -> List:
    """Given a folder path that contains a single or multiple '.tif' sequence of images files. Will create a folder with a similar name to the '.tif' file 
    and save the separate image frames in a 'movie' folder following the format accepted by MicroBundlePillarTrack (i.e. '*_####.TIF')."""
    main_paths_all = glob.glob(str(folder_path)+'/*.tif')
    main_paths_all = sorted(main_paths_all)
    new_paths_list = []
    for ii in range(len(main_paths_all)):
        input_file_str = main_paths_all[ii]
        file_name = os.path.basename(input_file_str)
        new_path = ia.create_folder(folder_path,file_name[:-4])
        new_paths_list.append(new_path)
        imgs_folder = ia.create_folder(new_path,'movie')
        im = imageio.mimread(input_file_str, memtest=False)
        count = 0
        for kk in range(0,len(im)):
            # save the images
            imageio.imwrite(str(imgs_folder) + '/%04d.TIF'%(kk + count), im[kk])
        count += 1
    return new_paths_list


