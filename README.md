# Microbundle Pillar Tracking Repository
MicroBundlePillarTrack

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


As with MicroBundleCompute, MicroBundlePillarTrack requires two main inputs: 1) two seperate binary masks for each of the microbundle pillars or posts and 2) consecutive movie frames of the beating microbundle. Within our pipeline, the pillar masks are gerenated automatically by either implementing a straightforward threshold based segmentation or by employing a fine-tuned version of the Segment [Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) [[1](#ref1)]. Nevertheless, we retain the option for the user to input a manually or externally generated mask in case both of our automated approaches fail. Following pillar segmentation, fiducial markers identified as Shi-Tomasi corner points are computed on the first frame of the movie and tracked across all frames. From this preliminary tracking, we can identify whether or not the microbundle movie starts from a fully relaxed position (valley frame), identify the first valley frame if it is different from frame 0 (first frame), and adjust the movie accordingly. From these tracked points, we are able to compute the pillars' mean directional and absolute displacements and convert these outputs into tissue force and tissue stress results. To visualize the results, the software outputs timeseries plots of the average tracked pillar displacements, tissue force, and tissue stress. 
<!-- Include temporal outputs: decay and velocity stuff -->

We are also adding new functionalities to the code as well as enhancing the software based on user feedback. Please check our [to-do list]((#todo)).

## Project Roadmap <a name="roadmap"></a>
In alignment with our long-term goal for developing open-source software for extracting mechanical information from mainly brightfield movies of beating cardiac microbundles (check [MicroBundleCompute](https://github.com/HibaKob/MicroBundleCompute) repository),   

MicroBundleCompute, 
The ultimate goal of this project is to develop and disseminate a comprehensive software for data curation and analysis from lab-grown cardiac microbundle on different experimental constructs. 

Prior to the initial dissemination of the current version, we have tested our code on approximately 30 examples provided by 2 different labs who implement different techniques. This allowed us to identify challenging examples for the software and improve our approach. We hope to further expand both our testing dataset and list of contributors.
The roadmap for this collaborative endeavor is as follows:

`Preliminary Dataset + Software` $\mapsto$ `Published Software Package` $\mapsto$ `Published Validation Examples and Tutorial` $\mapsto$ `Larger Dataset + Software Testing and Validation` $\mapsto$ `Automated Analysis of High-Throughput Experiments`

At present (**may 2023**), we have validated our software on a preliminary dataset in addition to a synthetically generated dataset (please find more details on the [SyntheticMicroBundle github page](https://github.com/HibaKob/SyntheticMicroBundle) and the [main manuscript](**add link**)). We also include details on validation against manual tracking [here](**add link to SA**). In the next stage, we are particularly interested in expanding our dataset and performing further software validation and testing. 
 Specifically, we aim to `1)` identify scenarios where our approach fails, `2)` create functions to accomodate these cases, and `3)` compare software results to previous manual approaches for extracting quantitative information, especially for pillar tracking. We will continue to update this repository as the project progresses.

## Installation Instructions <a name="install"></a>

## Tutorial <a name="tutorial"></a>

## Comparison to Available Tools <a name="comparison"></a>

## To-Do List <a name="todo"></a>
- [ ] Expand the test example dataset

## References to Related Work <a name="references"></a>
<a name="ref1"></a> [1] Kirillov, Alexander, et al. "Segment anything." arXiv preprint [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)(2023)


<!-- MicroBundleCompute Repo + paper
Dataset when up + microbundle dataset -->

## Contact Information <a name="contact"></a>
For additional information, please contact Emma Lejeune ``elejeune@bu.edu`` or Hiba Kobeissi ``hibakob@bu.edu``.

## Acknowledgements <a name="acknowledge"></a>
Thank you to Shoshana Das for providing the example tissue included with this repository. And -- thank you to Chad Hovey for providing templates for I/O, testing, and installation via the [Sandia Injury Biomechanics Laboratory](https://github.com/sandialabs/sibl) repository.