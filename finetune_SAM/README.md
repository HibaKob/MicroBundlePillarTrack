# Finetune Segment Anything Model (SAM)
We provide here the python script that we have implemented to finetune the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) and share a very small subset of our dataset that we have used for finetuning for "Type 1" data.

## Finetuning script
Given the ``finetune_microbundle_SAM_script.py`` script, finetuning SAM should be straightforward. The script can be run in a terminal with the conda environment ``microbundle-pillar-track-env`` activated. We recommend using GPUs but the code should work fine with CPUs.    

Only two parameters should be modified by the user: `1)` the desired path to which the output results should be saved ``results_path``, and `2` the path to the example image dataset and the ground truth masks, ``all_dataset_paths`` and ``all_mask_paths``, respectively. 

## Example dataset
In the ``dataset_example`` folder, we share an example training dataset of "Type 1". We organize our dataset as follows: