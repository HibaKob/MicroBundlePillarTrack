# Microbundle Pillar Track Repository

<!---
We will configure these once we make the repository public:

codecov_token: 1c09c4b5-8acb-4fc8-a222-c31983faab92

UPDATED functions in image_analysis.py:
compute_local_coverage
track_all_steps_with_adjust_param_dicts 
rotate_test_img (added outputs)

user warnings: 
1. low contrast movie: segmentation might fail
2. one pillar mask detected...taking second as the mirror: warning to examine results with caution
3. If segmented mask is smaller than min thresh area -> dilate + issue warning to user about that
4. subpixel displacements only: warning to examine results with caution
5. adjust movie to start from valley frame...print new frame0 number

reasons for analysis to fail:
1. no mask was created...skip example in run file
2. occasionally, if there is drift or displacements are subpixel and noisy, adjust_frmae_0 fails and tries to readjust adjusted movie...get OSError that directory is not empty: skip example in run file 

for both types of skipped examples, a text file is created with the examples that failed to run


deprecated warnings suppressed when testing 

-->

[![python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)


[![coverage_test](https://github.com/HibaKob/MicroBundlePillarTrack/actions/workflows/coverage_test.yml/badge.svg?branch=master)](https://github.com/HibaKob/MicroBundlePillarTrack/actions/workflows/coverage_test.yml) [![codecov](https://codecov.io/gh/HibaKob/MicroBundlePillarTrack/branch/master/graph/badge.svg?token=42GDTITVT0)](https://codecov.io/gh/HibaKob/MicroBundlePillarTrack)

## Table of Contents
* [Project Summary](#summary)
* [Project Roadmap](#roadmap)
* [Installation Instructions](#install)
* [Tutorial](#tutorial)
* [To-Do List](#todo)
* [References to Related Work](#references)
* [Contact Information](#contact)
* [Acknowledgements](#acknowledge)
<!-- * [Comparison to Available Tools](#comparison) -->

## Project Summary <a name="summary"></a>
The MicroBundlePillarTrack software is an adaptation of [MicroBundleCompute](https://github.com/HibaKob/MicroBundleCompute) software and is developed specifically for tracking the deformation of the pillars or posts of beating microbundles in brightfield movies. We consider two types of pillar-based microbundle platforms: `1)` "Type 1" which consist of standard experimental microbundle platforms termed microbundle strain gauges and `2)` "Type 2" experimental platforms which correspond to non-standard platforms termed FibroTUGs as described in detail in [[1](#ref1)] and [[2](#ref2)]. In this repository, we share the source code, steps to download and install the software, and tutorials on how to run its different functionalities. 


As with MicroBundleCompute, MicroBundlePillarTrack requires two main inputs: `1)` two separate binary masks for each of the microbundle pillars or posts and `2)` consecutive movie frames of the beating microbundle. Within our pipeline, the pillar masks are generated automatically by either implementing a straightforward threshold based segmentation or by employing a fine-tuned version of the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)[[3](#ref3)]. Nevertheless, we retain the option for the user to input a manually or externally generated mask in case both of our automated approaches fail. Following pillar segmentation, fiducial markers identified as Shi-Tomasi corner points are computed on the first frame of the movie and tracked across all frames. From this preliminary tracking, we can identify whether or not the microbundle movie starts from a fully relaxed position (valley frame), identify the first valley frame if it is different from frame 0 (first frame), and adjust the movie accordingly. By tracking the adjusted movie, we obtain pillar positions across consecutive frames, and subsequently, we are able to compute the pillars' mean directional and absolute displacements. Additional derived outputs include microbundle twitch force and stress results, and temporal outputs that include pillar contraction and relaxation velocities as well as full width (or duration) at half maximum (FWHM) and full width (or duration) at 80 maximum (FW80M).

<p align = "center">
<img alt="overview" src="tutorials/figs/Pillar_tracking_overview.png" width="100%" />

We are also adding new functionalities to the code as well as enhancing the software based on user feedback. Please check our [to-do list]((#todo)).

## Project Roadmap <a name="roadmap"></a>
In alignment with our long-term goal for developing open-source software for data curation and analysis from mainly brightfield movies of beating cardiac microbundles grown on different experimental constructs, we share here a tool for automated pillar analysis based on the validated [MicroBundleCompute](https://github.com/HibaKob/MicroBundleCompute) software. 

At this point (**January 2024**), we have tested our code on approximately 1,500 examples provided by 2 different labs who implement different techniques. This allowed us to identify challenging examples for the software and improve our approach. We hope to further expand both our testing dataset and list of contributors. We share the complete dataset and provide more details about it on [Dryad](provide_link_here){UPDATE LINK}.  

Specifically, through this collaborative endeavor we plan to proceed with the following roadmap:
`Preliminary Dataset + Software` $\mapsto$ `Published Software Package and Tutorial` $\mapsto$ `Preliminary Analysis of the Results` $\mapsto$ `Larger Dataset + Software Testing` $\mapsto$ `Statistical Model of Tissue Mechanical Behavior`

Looking forward, we are particularly interested in expanding our dataset and performing further software testing. 
Specifically, we aim to `1)` identify scenarios where our approach fails, `2)` accommodate these cases if possible, and `3)` identify and extract additional useful quantitative outputs. We will continue to update this repository as the project progresses.

## Installation Instructions <a name="install"></a>
### Get a copy of the microbundle pillar track repository on your local machine

To get started with using MicroBundlePillarTrack easily, we provide a recorded installation guide and a brief tutorial in this [demo](https://youtu.be/AigaLojbAYc). Alternatively, you can follow the steps below.

The best way to do this is to create a GitHub account and ``clone`` the repository. However, you can also download the repository by clicking the green ``Code`` button and selecting ``Download ZIP``. Download and unzip the ``MicroBundlePillarTrack-master`` folder and place it in a convenient location on your computer.

Alternatively, you can run the following command in a ``Terminal`` session:
```bash
git clone https://github.com/HibaKob/MicroBundlePillarTrack.git
```
Following this step, ``MicroBundlePillarTrack`` folder will be downloaded in your ``Terminal`` directory. 

### Create and activate a conda virtual environment

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) on your local machine.
2. Open a ``Terminal`` session (or equivalent) -- note that Mac computers come with ``Terminal`` pre-installed (type ``⌘-space`` and then search for ``Terminal``).
3. Type in the terminal to create a virtual environment with conda:
```bash
conda create --name microbundle-pillar-track-env python=3.9.13
```
4. Type in the terminal to activate your virtual environment:
```bash
conda activate microbundle-pillar-track-env
```
5. Check to make sure that the correct version of python is running (should be ``3.9.13``)
```bash
python --version
```
6. Update some base modules (just in case)
```bash
pip install --upgrade pip setuptools wheel
```

Note that once you have created this virtual environment you can ``activate`` and ``deactivate`` it in the future -- it is not necessary to create a new virtual environment each time you want to run this code, you can simply type ``conda activate microbundle-pillar-track-env`` and then pick up where you left off (see also: [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)).


### Install microbundle pillar track

1. Use a ``Terminal`` session to navigate to the ``MicroBundlePillarTrack-master`` folder or ``MicroBundlePillarTrack`` folder (depending on the method you followed to download the github repository). The command ``cd`` will allow you to do this (see: [terminal cheat sheet](https://terminalcheatsheet.com/))
2. Type the command ``ls`` and make sure that the file ``pyproject.toml`` is in the current directory.
3. Now, create an editable install of microbundle compute:
```bash
pip install -e .
```
4. If you would like to see what packages were installed, you can type ``pip list``

5. Download the finetuned SAM checkpoints for pillar mask segmentation from [here](https://doi.org/10.5281/zenodo.10642513) and save them in the ``src/microbundlepillartrack`` folder. It is critical that the downloaded checkpoints, ``microbundle_SAM_Type1_pillars.pth`` and ``microbundle_SAM_Type2_pillars.pth`` are saved in the same folder as the python script ``create_pillar_mask.py``. Note that you may need to unzip the downloaded files to obtain the desired `.pth` files. 

6. You can test that the code is working with pytest (all tests should pass):
```bash
pytest -v --cov=microbundlepillartrack --cov-report term-missing
```

As indicated by the shared badges above, the MicroBundlePillarTrack software has been tested on 3 operating systems, macOS, Windows, and Ubuntu with 97% coverage using GitHub Actions. We note that locally, we have tested the software on macOS Montery 12.7.1, Windows 10 22H2, and Ubuntu 20.04 with 100% coverage. Due to issues with loading model checkpoints with torch on Git LFS, we commented out 11 testing functions in the `test_create_pillar_mask.py` file found in the `tests` folder. For completeness and to check that the code is running 100% successfully, the user can uncomment these functions and rerun the command above. 

7. To run the code from the terminal, simply start python (type ``python``) and then type ``from microbundlepillartrack import image_analysis as ia``. For example:
```bash
(microbundle-pillar-track-env) hibakobeissi@Hibas-MacBook-Pro ~ % python
Python 3.9.13 (main, Oct 13 2022, 16:12:19) 
[Clang 12.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from microbundlepillartrack import image_analysis as ia
>>> ia.hello_microbundle_pillar_track()
'Hello World!'
>>> 
```

## Tutorial <a name="tutorial"></a>
We include within this repository a ``tutorials`` folder containing and a single example folder, 2 python scripts for running the code in single file mode or in batch mode, a python script to convert ``.tif`` image sequence files to individual ``.TIF`` frames, a python script to convert the software outputs saved as individual text files into ``.csv`` files, and finally, a python script to output a summary of the maximum and average results at maximum contraction or relaxation saved as a ``.csv`` file. To run the tutorial, change your current working directory to the ``tutorials`` folder.

### Data preparation 
The data (frames to be tracked) will be contained in the ``movie`` folder. Critically:
1. The files must have a ``.TIF`` extension.
2. The files can have any name, but in order for the code to work properly they must be *in order*. For reference, we use ``sort`` to order file names. By default, this function sorts strings (such as file names) alphabetically and numbers numerically. Below are examples of good and bad file naming practices. 

```bash
(microbundle-compute-env) hibakobeissi@Hibas-MacBook-Pro MicroBundleCompute-master % python
Python 3.9.13 (main, Oct 13 2022, 16:12:19) 
[Clang 12.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> bad_example = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
>>> bad_example.sort()
>>> print(bad_example)
['1', '10', '11', '12', '13', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9']
>>> 
>>> good_example = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]
>>> good_example.sort()
>>> print(good_example)
['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
>>>
>>> another_good_example = ["test_001","test_002","test_003","test_004","test_005","test_006","test_007","test_008","test_009","test_010","test_011","test_012","test_013","test_014","test_015"]
>>> another_good_example.sort()
>>> print(another_good_example)
['test_001', 'test_002', 'test_003', 'test_004', 'test_005', 'test_006', 'test_007', 'test_008', 'test_009', 'test_010', 'test_011', 'test_012', 'test_013', 'test_014', 'test_015']
```

3. The provided ``tif_sequence_to_TIFF_frames.py`` script can be used to prepare a batch of ``.tif`` image sequence files into the format accepted by ``MicroBundlePillarTrack``. To use the script, the data should be saved in a folder having the following original structure. As a side note, the ``files`` folder can have multiple ``.tif`` files but we include here a single example due to file size restrictions on GitHub. 
 <a name="data_prep"></a>
```bash
|___ files
        |___"tutorial_example.tif"
```

To run the provided script, simply do the following in a ``microbundle-pillar-track-env`` python terminal, where the variable ``input_folder`` is a [``PosixPath``](https://docs.python.org/3/library/pathlib.html) that specifies the relative path between where the code is being run (for example the provided ``tutorials`` folder) and the ``files`` folder that contains the ``.tif`` files to be analyzed.

```bash
(microbundle-pillar-track-env) hibakobeissi@Hibas-MacBook-Pro tutorials % python
Python 3.9.13 (main, Oct 13 2022, 16:12:19) 
[Clang 12.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from pathlib import Path
>>> import tif_sequence_to_TIFF_frames as tsf
>>> input_folder = Path('PATH_TO_FILES')
>>> tsf.tif_to_TIFF_frames(input_folder)
```

After running ``tif_sequence_to_TIFF_frames.py``, the folder structure should be similar to the example below, which is also the initial folder structure required for the pillar tracking code to work properly.

```bash
|___ files
        |___ tutorial_example
                    |___ movie
                            |___ *.TIF
        |___"tutorial_example.tif"
```

Aside from the folder structure, the code requires that the frames in the ``movie`` folder span at least 3 beats. We mandate this requirement for better result outputs. 

### Current core functionalities
In the current version of the code, there are $3$ core functionalities available for pillar tracking (automatic mask generation, tracking, and results visualization). As a brief note, it is not necessary to use all functionalities (e.g., you can still provide an externally generated mask and skip the automatic mask generation step or skip the visualization step).

 To be able to run the code, we stress that for the code snippets in this section, the variable ``input_folder`` is a [``PosixPath``](https://docs.python.org/3/library/pathlib.html), as defined [above](#data_prep), pointing to the folder that the user wishes to analyze.

 #### Automatic mask segmentation
 As mentioned [above](#summary), we base our automatic pillar mask segmentation functionality on two different approaches: `1)` a straightforward threshold-based approach and `2)` an AI-based approach that implements a fine-tuned version of the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)[[3](#ref3)]. 

The choice of segmentation approach is automatically determined by the code based on the user's input for the microbundle type. For "Type 1" input data, the code first performs segmentation based on local otsu thresholding. In case this first segmentation attempt fails to identify a mask for each pillar, a second trial at segmentation is done using the fine-tuned SAM for "Type 1" data. We adopt this two-trial approach because simple thresholding works well with most data examples of "Type 1", is significantly faster, and is less computationally expensive than SAM. "Type 2" data, on the other hand, present more challenging examples for a simple thresholding approach; instead, pillar segmentation is directly performed using fine-tuned SAM for "Type 2" data. 

Automatic pillar segmentation is performed by the function ``run_create_pillar_mask`` which requires 3 inputs: a [``PosixPath``](https://docs.python.org/3/library/pathlib.html) of the folder to be analyzed (``input_folder``), a [``PosixPath``](https://docs.python.org/3/library/pathlib.html) of the folder containing the fine-tuned SAM checkpoints (``checkpoint_path``), and the microbundle type (``microbundle_type``).

A straightforward method to run the segmentation step is to use the provided ``run_code_pillar.py`` or ``run_code_pillar_batch.py`` script files. The only difference between the two files is that ``run_code_pillar.py`` expects ``input_folder`` to point to a single data example folder containing a ``movie`` folder while ``run_code_pillar_batch.py`` expects ``input_folder`` to point to a folder containing several data example folders where each of these example folders contains a ``movie`` folder. In simpler terms, the former can be used to perform the analysis on a single file per run, while the latter script loops through all the examples within the main folder. An example input to either script files can be:

```bash
'''Indicate microbundle type'''   
microbundle_type = "type1" # Microbundle type can be either "type1" or "type2"

'''Indicate checkpoint path'''
checkpoint_path = Path('/LOCAL_PATH_TO_PACKAGE/MicroBundlePillarTrack/src/microbundlepillartrack')
```
Note that there is no need to provide ``input_folder`` when using this approach as it will be specified as a user input when calling the scripts as shown [below](#run_code).

 Alternatively, the user can run the pillar tracking functions directly in a ``microbundle-pillar-track-env`` python terminal as we show below.

```bash
from microbundlepillartrack import create_pillar_mask as cpm
from pathlib import Path

input_folder = Path('PATH_TO_FILES/tutorial_example')
microbundle_type = "type1"
checkpoint_path = Path('/LOCAL_PATH_TO_PACKAGE/MicroBundlePillarTrack/src/microbundlepillartrack')
cpm.run_create_pillar_mask(input_folder, checkpoint_path, microbundle_type)
```

Finally, if both segmentation approaches fail, the user can still provide externally generated pillar masks. To do this, first the ``cpm.run_create_pillar_mask(input_folder, checkpoint_path, microbundle_type)`` line in the script file should be commented out, and second, a binary text file for each pillar mask saved as ``pillar_mask_1.txt`` and ``pillar_mask_2.txt`` where the mask region is denoted by "1" and the background by "0", should be provided in the ``masks`` folder. An alternative approach would be to finetune the segment anything model (SAM) for your dataset in specific. In the ``finetune_SAM`` folder, we provide an example script to perform this task and share an example training dataset. 

#### Pillar tracking 
The function ``run_pillar_tracking`` will automatically read the data specified by the input folder (tiff files and mask file), run tracking, and save the results as text files.

The ``run_pillar_tracking`` expects a number of user-defined parameters: `1)` the tissue depth as estimated experimentally (``tissue_depth``), `2)` a unit vector specifying the orientation of the pillars (``pillar_orientation_vector``: this input is optional; if kept as `None`, the unit vector is automatically computed along the line connecting the two pillar centroids), `3)` the value of the pillar stiffness ($N/m$) (``pillar_stiffnes``) as measured experimentally, or `3a)` the pillar profile (``pillar_profile``) as either ``rectangular`` or ``circular``, `3b)` pdms Young's modulus (``pdms_E``) in MPa, `3c)` the pillar width (``pillar_width``) in $\mu m$, ``3d)`` the pillar thickness (``pillar_thickness``) in $\mu m$, ``3e)`` pillar diameter (``pillar_diameter``) in $\mu m$, ``3f)`` pillar length (``pillar_length``) in $\mu m$, and ``3g)``force application location  (``force_location``) in $\mu m$, two movie parameters ``4)`` the length scale (``ls``) in units of $\mu m$/pixel and ``5)`` the frames per second (``fps``), and finally `6)` a boolean (``split``) specifying if the tracking is to be carried out per beat (if ``True``) or per the entire movie (if ``False``). More details about this functionality are provided [below](#split).

We currently output all displacement results in units of pixels, force results in units of $\mu N$, velocity results in $\mu m$/s, tissue stress output in MPa, and time intervals with respect to frame number. We note that for calculating the pillar force and tissue stress, we follow the approach detailed in [[4](#ref4)] and elaborated on [below](#compute_stress), where the pillar is modeled as a cantilever. We are aware that different setups may have different pillar geometries and we plan to accommodate for this variation, as the need arises, in future iterations of the software. For calculating velocity outputs, we simply divide the displacement increments by the corresponding time intervals and implement a moving mean of window size 3 to smooth the resulting timeseries.  

```bash
from microbundlepillartrack import pillar_analysis as pa
from pathlib import Path

'''Movie parameters: frames per second(fps) and length scale (ls) as micrometers/pixel'''
fps = 30 
ls = 4

'''Indicate microbundle type'''   
microbundle_type = "type1" # Microbundle type can be either "type1" or "type2"
'''Pillar stiffness can be directly provided: replace `None` by a value''' 
pillar_stiffnes = 2.677 # Provide this value in Newton per meter (N/m) 
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
tissue_depth = 350 # Provide this value in micrometer (um) (type 1)

''' Set to `True` to eliminate drift if observed in pillar results'''
split = False

pa.run_pillar_tracking(input_folder, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pdms_E, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, ls, fps, split)
```

#### Post-tracking visualization
The function ``visualize_pillar_tracking`` is for visualizing the pillar tracking results consisting of timeseries plots of pillar displacement, force, and contraction and relaxation velocities, in addition to tissue stress, as shown [below](#results). It takes length scale (``ls``) in units of $\mu m$/pixel, frames per second (``fps``), and (``split``) boolean as inputs. We note that in a continuous session, there is no need to redefine these parameters or to import the needed packages again. We include them in the example below for the sake of completeness only. 

```bash
from microbundlepillartrack import pillar_analysis as pa
from pathlib import Path

'''Movie parameters: frames per second(fps) and length scale (ls) as micrometers/pixel'''
fps = 30 
ls = 4

''' Set to `True` to eliminate drift if observed in pillar results'''
split = False

pa.visualize_pillar_tracking(input_folder, ls, fps, split)
```

As demonstrated here, the entire tracking and visualization process is fully automated and requires very little input from the user.

#### First valley adjustment
To ensure that tracking is performed with respect to a fully relaxed tissue (i.e. a valley frame), we implement, within our computational pipeline, two functions, ``check_frame0_valley`` and ``adjust_first_valley``, that check whether or not the input movie begins at a valley frame and automatically adjust the movie to begin at one, if required. A user warning is issued when the movie is adjusted specifying the new starting frame. The adjusted frame list is saved into the ``movie`` folder while the original frame files are retained in the ``unadjusted_movie`` folder. 

#### Drift correction <a name="split"></a>
From our extensive experience analyzing mainly brightfield movies of beating microbundles, a number of examples display non-trivial drift in the tracked results mainly due to imaging noise. We remove this drift by performing temporal segmentation into individual peaks, where the first frame of each beat is considered to be the reference for all output calculations within the beat. 

We note that drift correction is not performed automatically; instead, a warning is issued to the user who can later decide whether or not to repeat tracking with beat splitting by setting the ``split`` input parameter to ``True``. In some cases, high drift is caused by imaging artifacts such as shadows or degraded image quality that cannot be eliminated by splitting. When this happens, a warning is given to the user that drift is still present in the tracked results and that it is recommended to analyze only a portion of the movie. It is critical that a minimum of 3 beats is present in any movie to enable automatic analysis and carrying out the necessary adjustments and corrections. 

#### Tissue stress computation  <a name="compute_stress"></a>
As aforementioned, twitch force, defined as indicated in the schematic [below](#force_exp_fig), is calculated by modeling the pillars or posts as cantilevers [[4](#ref4)]. Tissue stress is then obtained by normalizing the mean of the two twitch forces obtained for each pillar by the cross-sectional area of the microbundle (depth $x$ width). To find this cross-sectional area, the user is required to provide the microbundle depth in $\mu m$. The microbundle width is calculated automatically by thresholding a region of the first frame centered around the microbundle, taking the average of the resulting binary mask along the column direction, and finally calculating the distance between the sharpest two changes in mean intensity indicating the transition from a dark background to a light tissue region and then back to a dark background. We note that microbundle width calculation is performed on a rotated frame such that a vector joining the centroids of the two pillars aligns with the horizontal direction. 

For suboptimal examples, low contrast videos for instance, the calculated microbundle width may not be accurate. In these cases, a warning notifies the user of this possibility. Also, a binary mask of the region used for width calculation is saved to the ``masks`` folder as ``mid_tissue_mask.png`` to enable the user to assess the accuracy of the automated width estimation. 

 <a name="force_exp_fig"></a>
<p align = "center">
<img alt="overview" src="tutorials/figs/Pillar_force_schematic_v3.png" width="100%" />

### Running the code  <a name="run_code"></a>
Once the software is [installed](#install) and the data is set up according to the [instructions](#data_prep), running the code is quite straightforward. To run the tutorial example, navigate in the Terminal so that your current working directory is in the ``tutorials`` folder. To run the code on the provided single example, type:

```bash
python run_code_pillar.py files/tutorial_example
```

And it will automatically run the example specified by the ``files/tutorial_example`` folder and the associated visualization function. You can use the ``run_code_pillar.py`` to run your own code, you just need to specify a relative path between your current working directory (i.e., the directory that your ``Terminal`` is in) and the data that you want to analyze. Alternatively, you can modify ``run_code_pillar.py`` to make running the code more convenient (i.e., remove command line arguments, skip some steps). Here is how the outputs of the code will be structured (in the same folder as input ``movie``):

```bash
|___ example_data
|        |___ movie
|                |___"*.TIF"
|        |___ unadjusted_movie       (if original movie is adjusted)
|                |___"*.TIF"
|        |___ masks
|                |___"pillar_mask_1.txt"      
|                |___"pillar_mask_1.png"         
|                |___"pillar_mask_2.txt"
|                |___"pillar_mask_2.png"        
|                |___"pillar_masks_overlay.png" 
|                |___"mid_tissue_mask.png" 
|        |___ pillar_results
|                |___"orientation_info.txt"
|                |___"pillar%i_beat_peaks.txt"
|                |___"pillar%i_info.txt"
|                |___"pillar%i_row.txt"                     (if split = False)
|                |___"pillar%i_col.txt"                     (if split = False)
|                |___"pillar%i_beat%i_row.txt"              (if split = True)
|                |___"pillar%i_beat%i_col.txt"              (if split = True)
|                |___"pillar%i_pillar_velocity.txt"
|                |___"pillar%i_pillar_contraction_vel_info.txt"
|                |___"pillar%i_pillar_relaxation_vel_info.txt"
|                |___"pillar%i_t50_beat_width_info.txt"
|                |___"pillar%i_t80_beat_width_info.txt"
|                |___"pillar%i_pillar_force_abs.txt"
|                |___"pillar%i_pillar_force_row.txt"
|                |___"pillar%i_pillar_force_col.txt"
|                |___"tissue_width_info.txt"
|                |___"tissue_stress.txt"
|        |___ pillar_visualizations
|                |___"pillar_directional_displacement.pdf"
|                |___"pillar_mean_absolute_displacement.pdf"
|                |___"pillar_time_intervals.pdf"
|                |___"pillar_velocity_results.pdf"
|                |___"pillar_force_absolute.pdf"
|                |___"tissue_stress.pdf"
```

We note that we systematically denote the left pillar as ``pillar1`` and the right one as ``pillar2`` and accordingly, ``pillar_mask_1`` corresponds to the left pillar while ``pillar_mask_2`` corresponds to the right one.

### Understanding the output files
The outputs of running the software will be stored in the ``pillar_results`` folder as text files. This folder contains 8 ``"info"`` text files: ``orientation_info.txt``, ``pillar%i_beat_peaks.txt``, ``pillar%i_info.txt``, ``pillar%i_pillar_contraction_vel_info.txt``, and ``pillar%i_pillar_relaxation_vel_info.txt``, ``pillar%i_t50_beat_width_info.txt``, ``pillar%i_t80_beat_width_info.txt``, and ``tissue_width_info.txt``.


The ``orientation_info.txt`` file stores information regarding the orientation of the unit vector connecting the pillar centroids (left pillar to the right one) with respect to the horizontal direction. It has three columns. Column 0 refers to the horizontal component of the direction vector while column 1 refers to its vertical component assuming that the origin of the orthonormal system is the bottom left corner of the frame. Column 2 refers to the angle (in radians) that this direction vector makes with the horizontal, with positive angles taken in the counterclockwise direction. For example, ``orientation_info.txt`` could contain:
```bash
1 0 0
```

This means that the unit vector along the line connecting the pillar centroids aligns with the horizontal direction (1,0) and as such makes an angle of 0 with it. 

The ``pillar%i_beat_peaks.txt`` file contains the frame numbers at which beat peaks are identified and is outputted for each pillar. These frame numbers are in reference to the adjusted movie (or alternatively the original movie if it starts from a valley frame) 0 indexed. The file would have as many rows as the number of identified peaks. For example, ``pillar1_beat_peaks.txt`` could contain:
```bash
16
40
64
88
```
This means that the first identified beat peak occurs at frame 16, the second beat peak occurs at frame 40, and so on. 

The ``pillar%i_info.txt`` file stores information pertaining to beat identification and segmentation and is unique for each pillar. It has three columns. Column 0 refers to the ``beat_number`` (e.g., beat 0 is the first beat, beat 1 is the second beat etc.). Column 1 refers to the ``first_frame`` of each beat. Column 2 refers to the ``last_frame`` of each beat. These will be the frame numbers in the adjusted movie (or alternatively the original movie if it starts from a valley frame) 0 indexed. For example, ``pillar1_info.txt`` could contain:

```bash
0 3 25
1 25 49
2 49 72
```

This means that beat 0 starts at frame 3 and ends at frame 25, beat 1 starts at frame 25 and ends at frame 49, and beat 2 starts at frame 49 and ends at frame 72. Note that if full beats cannot be segmented from the timeseries data there may be issues with running the code. Often, the visually apparent first and last beats will be excluded from this segmentation because we cannot identify clear start and end points.


The files ``pillar%i_pillar_contraction_vel_info.txt`` and ``pillar%i_pillar_relaxation_vel_info.txt`` contain the frame numbers with respect to the adjusted movie (or alternatively the original movie if it starts from a valley frame) 0 indexed, at which the maximum (peak) contraction or relaxation velocity occur respectively, for each pillar. Each file would have as many rows as the number of identified contraction or relaxation maxima. For example, either ``pillar1_pillar_contraction_vel_info.txt`` or ``pillar1_pillar_relaxation_vel_info.txt`` could contain:

```bash
12
36
60
84
```

This means that peak velocities, while the microbundle is contracting or relaxing, occur at frames 12, 36, 60, and 84.

The files ``pillar%i_t50_beat_width_info.txt`` and ``pillar%i_t80_beat_width_info.txt`` store information pertaining to the full width at half maximum (FWHM) and full width at 80 maximum (FW80M), respectively, for each pillar. Each of the two files contains 4 rows. Row 0 refers to the ``full_width`` of each beat at either half maximum or 80 maximum. Row 1 refers to ``width_heights``, the value of the height (magnitude) of the mean absolute displacement at which the widths were evaluated in pixels. Rows 2 and 3 refer to the ``left_ips`` and ``right_ips``, the interpolated positions of the left and right intersection points of a horizontal line at the evaluation height, respectively. We note that the values of widths and interpolated positions are given as frame numbers with respect to the adjusted movie (or alternatively the original movie if it starts from a valley frame) 0 indexed. Each of these files is expected to have dimension ``4 x BB``, where BB corresponds to the number of beats. For example, the ``pillar1_t50_beat_width_info.txt`` or  ``pillar1_t80_beat_width_info.txt`` file could contain:

```bash
5.9 5.8 5.9 5.9
0.6 0.5 0.5 0.5
13.3 37.4 61.3 85.7
19.2 43.2 67.2 91.6
```

This means that for the 4 identified beats, the full width at either 50 or 80 maximum (depending on the text file being examined) is 5.9 frames for beat 0, 5.8 frames for beat 1, and 5.9 frames for beats 2 and 3 respectively. The value of the mean absolute displacement at which each of these widths was evaluated are 0.6, 0.5, 0.5, and 0.5 pixels respectively. Finally, a horizontal line extended at the evaluation height intersects beat 0 at frames 13.3 and 19.2, corresponding to a full width of 5.9 frames, beat 1 at frames 37.4 and 43.2 (resulting in a full width of 5.8 frames), beat 2 at frames 61.3 and 67.2 (resulting in a full width of 5.9 frames), and beat 3 at frames 85.7 and 91.6 (resulting in a full width of 5.9 frames).  

Finally, the ``tissue_width_info.txt`` contains two values corresponding to the tissue width (in the row direction) measured at the center of the tissue mask. The first value is saved in units of pixels while the second one is in $\mu m$.

Aside from the ``info`` text files, the ``"row"`` and ``"col"`` files contain information regarding the row and column positions of the tracked marker (fiducial) points. For default pillar tracking without time segmentation, there will be one row-position file and one col-position file for the entire movie for each tracked pillar. Specifically:
* ``pillar%i_row.txt`` will contain the image row positions of each marker for the pillar specified by ``%i``
* ``pillar%i_col.txt`` will contain the image column positions of each marker for the pillar specified by ``%i``
However, if the optional time segmentation step is run to remove any drift present in the tracked pillar results, there will be one row-position file and one col-position file for each beat as follows: 
* ``pillar%i_beat%i_row.txt`` will contain the image row positions of each marker for the pillar specified by ``%i`` and beat ``%i``
* ``pillar%i_beat%i_col.txt`` will contain the image column positions of each marker for the pillar specified by ``%i``  and beat ``%i``

In these text files, the rows correspond to individual markers, while the columns correspond to the frames. For example, if a file has dimension ``AA x BB``, there will be ``AA`` markers and ``BB`` frames. For tracking without time segmentation, the number of frames will correspond to the total number of frames tracked, while for tracking with time segmentation, the number of frames will be the number of frames per beat. 

The ``pillar%i_pillar_force_abs.txt``, ``pillar%i_pillar_force_row.txt``, and ``pillar%i_pillar_force_col.txt`` files will store results (in $\mu N$) corresponding to the mean absolute pillar force, mean pillar force in the row direction, and mean pillar force in the column direction respectively for each pillar. The files store timeseries results and have lengths equal to the number of input movie frames when time segmentation is skipped. Alternatively, if time segmentation is implemented, the timeseries results will have a length equal to the number of frames corresponding to the tracked beats (i.e. excluding the first and last beats). 

Similarly, the ``tissue_stress.txt`` file will store the calculated tissue stress (in MPa) based on the average mean absolute force of the two pillars. Again, the file stores a timeseries result of length equal to the number of input movie frames when time segmentation is skipped, and a length equal to the number of frames corresponding to the tracked beats (i.e. excluding the first and last beats) if time segmentation is implemented. 

Finally, the ``pillar%i_pillar_velocity.txt`` file stores the rate of contraction and relaxation of the microbundle in units of $\mu m$/s, as computed at each pillar. The length of the timeseries result stored in this file is equal to the number of input movie frames minus the moving mean window size, when time segmentation is skipped. Alternatively, if time segmentation is implemented, the timeseries results will have a length equal to the number of frames corresponding to the tracked beats (i.e. excluding the first and last beats) minus the moving mean window size. For example, if 200 frames are tracked and the moving mean window size is 3, the length of the velocity timeseries will be 197.  

For convenience, we also include 2 python scripts to ``1)`` convert individual ``.txt`` result files into individual ``.csv`` files and ``2)`` generate a summary ``.csv`` file of the maximum and average results at maximum contraction or relaxation. These 2 scripts can be run in a manner similar to running ``tif_sequence_to_TIFF_frames.py`` described [above](#data_prep); just make sure to provide the correct inputs to each function.

### Understanding the visualization results <a name="results"></a>
The outputs of running the visualization code will be stored in the ``pillar_visualizations`` folder. Six timeseries plots should be contained here: ``pillar_directional_displacement.pdf``, ``pillar_mean_absolute_displacement.pdf``, ``pillar_force_absolute.pdf``, ``tissue_stress.pdf``, ``pillar_time_intervals.pdf``, and ``pillar_velocity_results.pdf``. These plots correspond to the variation of the pillar mean row and column displacements, pillar mean absolute displacement, pillar absolute force, and mean tissue stress with respect to frame number, as well as full width at half and 80 maximum and pillar contraction and relaxation velocity with respect to time (s), respectively. We note that the results of the two tracked pillars will be visualized on the same plots except for ``tissue_stress`` which is taken as the average of the two pillar results. We include below 4 example output plots. 

<p align = "left">
<img alt="pillar directional displacement visualization" src="tutorials/files/figs/visualizations/pillar_directional_displacement.png" width="48%" />
&nbsp;
<img alt="pillar stress" src="tutorials/files/figs/visualizations/tissue_stress.png" width="48%"/>
</p>

<p align = "center">
<img alt="pillar time intervals" src="tutorials/files/figs/visualizations/pillar_time_intervals.png" width="37.3%"/>
&nbsp;
<img alt="pillar velocity" src="tutorials/files/figs/visualizations/pillar_velocity_results.png" width="45%"/>
</p>

In all cases, the output visualizations are stored as ``.pdf`` files for higher resolution.

<!-- ## Comparison to Available Tools <a name="comparison"></a> -->

## To-Do List <a name="todo"></a>
- [ ] Expand the test example dataset
- [ ] Explore options for additional analysis/visualization

## References <a name="references"></a>
<a name="ref1"></a> [1] Kobeissi, Hiba et al. (2024). MicroBundleCompute: Automated segmentation, tracking, and analysis of subdomain deformation in cardiac microbundles. PLOS One. https://doi.org/10.1371/journal.pone.0298863

<a name="ref2"></a> [2] Kobeissi, Hiba et al. (2023). Engineered cardiac microbundle time-lapse microscopy image dataset [Dataset]. Dryad. https://doi.org/10.5061/dryad.5x69p8d8g

<a name="ref3"></a> [3] Kirillov, Alexander, et al. (2023). Segment anything. arXiv preprint [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

<a name="ref4"></a> [4] Legant, Wesley R. et al. (2009). Microfabricated tissue gauges to measure and manipulate forces from 3D microtissues. Proceedings of the National Academy of Sciences, 106(25), 10097-10102. https://doi.org/10.1073/pnas.0900174106

Related repositories include:
* https://github.com/HibaKob/MicroBundleCompute

## Contact Information <a name="contact"></a>
For additional information, please contact Emma Lejeune ``elejeune@bu.edu`` or Hiba Kobeissi ``hibakob@bu.edu``.

## Acknowledgements <a name="acknowledge"></a>
Thank you to Shoshana Das and Samuel DePalma for providing the example tissues included with this repository. And -- thank you to Chad Hovey for providing templates for I/O, testing, and installation via the [Sandia Injury Biomechanics Laboratory](https://github.com/sandialabs/sibl) repository.