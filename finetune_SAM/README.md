# Finetune Segment Anything Model (SAM)
We provide here the python script that we have implemented to finetune the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) and share a very small subset of our dataset that we have used for finetuning for "Type 1" data.

## Finetuning script
Given the ``finetune_microbundle_SAM_script.py`` script, finetuning SAM should be straightforward. The script can be run in a terminal with the conda environment ``microbundle-pillar-track-env`` activated. We recommend using GPUs but the code should work fine with CPUs.    

For the script to run, only 3 parameters are necessary to be modified by the user: `1)` the desired path to which the output results should be saved ``results_path``, `2` the path to the example image dataset and the ground truth masks, ``all_dataset_paths`` and ``all_mask_paths``, respectively, and `3` the path to the original ViT-B SAM model checkpoint `sam_vit_b_01ec64.pth` which can be obtained [here](https://github.com/facebookresearch/segment-anything).

## Example dataset
In the ``dataset_example`` folder, we share an example training dataset of "Type 1". We prepare our dataset as follows: 
* Isolate around 15 - 20 frames from a `tif` image sequence of 300 or more frames where the pillars appear to be relatively in the same location, yet the microbundle is beating such that the frames are distinctive. The files ``Example_01.tif`` and ``Example_02.tif`` are example outputs od this step.
* Generate a single binary mask for both pillars and create a repetitive `tif` sequence of length equal to the number of isolated frames. The files ``mask_Example_01.tif`` and ``mask_Example_02.tif`` are example outputs od this step.
* Repeat the previous two steps for each sequence in the training dataset separately. We do not merge the different files as the image sizes between the different examples may vary. Different image sizes will be automatically unified later within the provided python script. 

In total, finetuning was performed with 275 images obtained as described above from only 11 `tif` sequence examples of "Type 1". On the other hand, "Type 2" microbundle data is more challenging and finetuning was performed with 1060 images obtained from 53 `tif` sequence examples. Critically, to increase the diversity of the dataset, we implemented a random rotation for each image and its associated ground truth mask. This is performed automatically within ``finetune_microbundle_SAM_script.py`` file. 