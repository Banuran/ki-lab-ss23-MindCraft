# HKA KI-Lab SS23 Group MindCraft

This Repository accompanies the project phase of the KI-Lab.

## Data

The data must be saved in the data folder. The data is available here https://raganato.github.io/vwsd/. The datatsets for train and trial can be simply downloaded and extracted. For the test data we choose to use the resized dataset. Therefore extracted test images must be saved to ./data/test_images_resized/, while the gold labels directly reside in the ./data/ folder. 

## Baseline

Our two baseline models can be found in the directory with the same name. We wrote our own version of the CLIP basline used in the paper. Additionally we implemented an imporved version using the more performant SigLIP model.

## Files

## Dataset

We wrote two jupyter notebooks to familiarize ourselvs with the provided data.

- data-analysis.ipynb: This notebook gives a small glimpse into the training and test data.
- data-loader.ipynb: This notebook was primarily used to develop our pytorch dataset class. It also contains example outputs of said class and demonstrates all features of the implemented dataset.

The finished dataset class resided in the dataset.py file. This file then gets imported into our model classes for loading the training or test data. 

The dataset can return training, test or validation data. It is possible to select the language of the returned set and set image as well as text transformations to be applied to the images. Theres also an option to enable the data augmentation used by us. Additionally an translated version of the farsi and italian version can be selected.
The file also contains a sconde dataset class used for our work with the Flicker30K dataset.
