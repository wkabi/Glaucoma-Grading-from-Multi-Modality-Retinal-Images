# Glaucoma-Grading-from-Multi-Modality-Retinal-Data
This repository will help you to train, validate and test Multimodal AI model for Glaucoma Grading using retinal fundus and OCT images by employing deep learning (DL) methodology. 

Important Notes: 
- This repository is not for commercial use/purpose. 
- It can be used for research purpose (only).
- If you utilize this repository that will lead to any publication, please cite this repository.
- If you utilize this repository that will lead to any publication, please cite the following papers those are most relevant. Details can be found from my Google Scholar page (https://scholar.google.com/citations?user=3rrT0vgAAAAJ&hl=en).
1. IDF21-0299 An automatic deep learning-based system for screening and management of DME.
2. Automatic screening and progress with AI-assisted OCT in retinal macular oedema detection.
3. A domain adaptation method for deep learning based automatic diabetic retinopathy grading.
4. Exploring the transferability of a foundation model for fundus images: Application to hypertensive retinopathy.
5. Gamma challenge: glaucoma grading from multi-modality images.
6. An AI-assisted system for screening of diabetic macular edema (DME) by using real-world fundus images
7. An AI-assisted system for early screening of age-related macular degeneration (ARMD) from real-world fundus images
8. An AI-assisted system for classification and triage of diabetic retinopathy (DR) by using real-world fundus images.
9. Hybrid classification of hypertensive retinopathy (HR) in general population based on artificial intelligence
10. IDF23-0249 Hybrid Classification of Hypertensive Retinopathy (HR) in General Population Based on Artificial Intelligence
11. IDF23-0251 An AI-Assisted System for Classification and Triage of Diabetic Retinopathy (DR) by Using Real-World Fundus Images
12. Domain generalization for diabetic retinopathy grading through vision-language foundation models
13. All that glitters is not gold: are current retina foundation models able to efficiently detect hypertensive retinopathy?
14. AI-assisted automated screening of retinal anomalies in OCT Images: A deep learning approach
15. Parameter-efficient fine-tuning of ophthalmology foundation models for robust hypertensive retinopathy detection and triage
- Finally, don't forget to hit the 'star' button for this repository :)

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

