# Log changes

This file tracks all of the changes made to the original codebase to run on the supercloud cluster, where the compute nodes don't have access to internet

## Downloading the pretrained models 

The clip models are downloaded or cached during runs.
Our solution is to download them before the run, and to store them in `clip/models`
The script doing this is in `label-free-cbm/utils`