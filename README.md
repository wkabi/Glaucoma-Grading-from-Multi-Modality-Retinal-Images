# Optic Glaucomma grading

This project is developed with the configuring framework : [Hydra](https://hydra.cc/) .

We also track the experimental results with [wandb](https://wandb.ai/) .


# Install

Main depencencies
```
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
segmentation-models-pytorch
albumentations
hydra-core
wandb
```
See requirements.txt for the full enviroments.

Then setup
```
python setup.py develop
```

# Data preparation
After downloading the dataset, please use `tools/convert_fundusdataset.py` to generate the h5 data files. This is to accelerate the data loading during model training and testing.
Then you need to change the `data_root` in the yaml files under the `configs/data` folder to the absolute path of your data root path in your machine before you run some jobs.

For getting the results on the test set, a fake annotation csv file named `all.csv` is required by the code. Just set the labels for all the samples as -1.

# Test
Download the models from : [LINK TO BE GIVEN]

See the `test_pipeline.sh` about how to get our results for the GAMMA contest. 
Remeber to modify the path in the script before you run it.


# Train
See the `train_multimodal.sh` for model training.

Remeber to modify the `data_root` in the data config file `configs/data/fundus_grade_oct.yaml` to the absolute data path on your machine before run it.

