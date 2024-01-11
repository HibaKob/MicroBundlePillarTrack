# Microbundle Pillar Track Repository

<!---
We will configure these once we make the repository public:

codecov_token: 62ae94e3-b311-42ed-a9ef-0a280f77bd7c

[![python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![tests](https://github.com/hibakob/microbundlecompute/workflows/coverage_test/badge.svg)](https://github.com/hibakob/microbundlecompute/actions) [![codecov](https://codecov.io/gh/hibakob/microbundlecompute/branch/master/graph/badge.svg?token=EVCCPWCUE7)](https://codecov.io/gh/hibakob/microbundlecompute)


UPDATED functions in image_analysis.py:
compute_local_coverage
track_all_steps_with_adjust_param_dicts 
rotate_test_img (added outputs)

change from microbundlecompute import.... to from microbundlepillartrack import...

NOTE: pillar_1 is either left pillar or if 2 pillars failed to be segmented then it is the one that was automatically identified

WT_DMSO_2_5: mask of pillar 1 is off (it isn’t visible in the original video): “thresh_size_large” for area<9000 gets rid of wrong pillar 1 (Interesting observation: pillar 1 mask is detected but has higher eccentricity so the region is discarded) — Changes to code: Get 4 roundest regions, sort in order of decreasing area, and get largest 2

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

## Table of Contents
* [Project Summary](#summary)
* [Project Roadmap](#roadmap)
* [Installation Instructions](#install)
* [Tutorial](#tutorial)
* [Comparison to Available Tools](#comparison)
* [To-Do List](#todo)
* [References to Related Work](#references)
* [Contact Information](#contact)
* [Acknowledgements](#acknowledge)

## Project Summary <a name="summary"></a>
The MicroBundlePillarTrack software is an adaptation of [MicroBundleCompute](https://github.com/HibaKob/MicroBundleCompute) software and is developed specifically for tracking the deformation of the pillars or posts of beating microbundles in brightfield movies. In this repository, we share the source code, steps to download and install the software, and tutorials on how to run its different functionalities. 


As with MicroBundleCompute, MicroBundlePillarTrack requires two main inputs: 1) two seperate binary masks for each of the microbundle pillars or posts and 2) consecutive movie frames of the beating microbundle. Within our pipeline, the pillar masks are gerenated automatically by either implementing a straightforward threshold based segmentation or by employing a fine-tuned version of the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) [[1](#ref1)]. Nevertheless, we retain the option for the user to input a manually or externally generated mask in case both of our automated approaches fail. Following pillar segmentation, fiducial markers identified as Shi-Tomasi corner points are computed on the first frame of the movie and tracked across all frames. From this preliminary tracking, we can identify whether or not the microbundle movie starts from a fully relaxed position (valley frame), identify the first valley frame if it is different from frame 0 (first frame), and adjust the movie accordingly. By tracking the adjusted movie, we obtain pillar positions across consecutive frames, and subsequently, we are able to compute the pillars' mean directional and absolute displacements. Additional derived outputs include microbundle twitch force and stress results, and temporal outputs that include pillar contraction and relaxation velocities as well as full width (or duration) at half maximum (FWHM) and full width (or duration) at 90 maximum (FW90M).

We are also adding new functionalities to the code as well as enhancing the software based on user feedback. Please check our [to-do list]((#todo)).

## Project Roadmap <a name="roadmap"></a>
In alignment with our long-term goal for developing open-source software for data curation and analysis from mainly brightfield movies of beating cardiac microbundles grown on different experimental constructs, we share here a tool for automated pillar analysis based on the validated [MicroBundleCompute](https://github.com/HibaKob/MicroBundleCompute) software. 

At this point (**January 2024**), we have tested our code on approximately 700 examples provided by 2 different labs who implement different techniques. This allowed us to identify challenging examples for the software and improve our approach. We hope to further expand both our testing dataset and list of contributors. We share the complete dataset and provide more details about it on [Dryad](provide_link_here){UPDATE LINK}.  

Specifically, through this collaborative endeavor we plan to proceed with the following roadmap:
`Preliminary Dataset + Software` $\mapsto$ `Published Software Package and Tutorial` $\mapsto$ `Preliminary Analysis of the Results` $\mapsto$ `Larger Dataset + Software Testing` $\mapsto$ `Statistical Model of Tissue Mechanical Behavior`

Looking forward, we are particularly interested in expanding our dataset and performing further software testing. 
Specifically, we aim to `1)` identify scenarios where our approach fails, `2)` accomodate these cases if possible, and `3)` identify and extract additional useful quantitative outputs. We will continue to update this repository as the project progresses.

## Installation Instructions <a name="install"></a>
### Get a copy of the microbundle pillar track repository on your local machine

The best way to do this is to create a GitHub account and ``clone`` the repository. However, you can also download the repository by clicking the green ``Code`` button and selecting ``Download ZIP``. Download and unzip the ``MicroBundlePillarTrack-main`` folder and place it in a convenient location on your computer.

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

1. Use a ``Terminal`` session to navigate to the ``MicroBundlePillarTrack-main`` folder or ``MicroBundlePillarTrack`` folder (depending on the method you followed to download the github repository). The command ``cd`` will allow you to do this (see: [terminal cheat sheet](https://terminalcheatsheet.com/))
2. Type the command ``ls`` and make sure that the file ``pyproject.toml`` is in the current directory.
3. Now, create an editable install of microbundle compute:
```bash
pip install -e .
```
4. If you would like to see what packages were installed, you can type ``pip list``
5. You can test that the code is working with pytest (all tests should pass):
```bash
pytest -v --cov=microbunlepillartrack  --cov-report term-missing
```
6. To run the code from the terminal, simply start python (type ``python``) and then type ``from microbundlepillartrack import image_analysis as ia``. For example:
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
We include within this repository a ``tutorials`` folder containing and a single example folder, 2 python scripts for running the code in single file mode or in batch mode, a python script to convert ``.tif`` image sequence files to individual ``.TIF`` frames, and a python script to convert the software outputs saved as individual text files into a single ``.csv`` file . To run the tutorial, change your curent working directory to the ``tutorials`` folder.

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

To run the provided script, simply do the following in a terminal running python, where the variable ``input_folder`` is a [``PosixPath``](https://docs.python.org/3/library/pathlib.html) that specifies the relative path between where the code is being run (for example the provided ``tutorials`` folder) and the ``files`` folder that contains the ``.tif`` files to be analyzed, as defined [above](#data_prep).


```bash
(microbundle-pillar-track-env) hibakobeissi@Hibas-MacBook-Pro tutorials % python
Python 3.9.13 (main, Oct 13 2022, 16:12:19) 
[Clang 12.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from pathlib import Path
>>> import tif_sequence_to_TIFF_frames as tsf
>>> input_folder = Path('PATH_TO_FILES')
>>> tsf.tif_to_TIFF_frames(input_folder)
[PosixPath('PATH_TO_FILES/tutorial_example')]
>>> 
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
In the current version of the code, there are 3 core functionalities available for pillar tracking (automatic mask generation, tracking, and results visualization). As a brief note, it is not necessary to use all functionalities (e.g., you can still provide an externally generated mask and skip the automatic mask generation step or skip the visualization step).

 To be able to run the code, we stress that for the code snippets in this section, the variable ``input_folder`` is a [``PosixPath``] as defined [above](#data_prep) that the user wishes to analyze.
 
## Comparison to Available Tools <a name="comparison"></a>

## To-Do List <a name="todo"></a>
- [ ] Expand the test example dataset
- [ ] Explore options for additional analysis/visualization

## References to Related Work <a name="references"></a>
<a name="ref1"></a> [1] Kirillov, Alexander, et al. "Segment anything." arXiv preprint [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)(2023)


<!-- MicroBundleCompute Repo + paper
Dataset when up + microbundle dataset -->

## Contact Information <a name="contact"></a>
For additional information, please contact Emma Lejeune ``elejeune@bu.edu`` or Hiba Kobeissi ``hibakob@bu.edu``.

## Acknowledgements <a name="acknowledge"></a>
Thank you to Shoshana Das and Samuel DePalma for providing the example tissues included with this repository. And -- thank you to Chad Hovey for providing templates for I/O, testing, and installation via the [Sandia Injury Biomechanics Laboratory](https://github.com/sandialabs/sibl) repository.