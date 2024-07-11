# HKA KI-Lab SS23 Group MindCraft

This Repository accompanies the project phase of the KI-Lab. The task was to solve the SemEval-2023 Visual Word Sense Disambiguation (Visual-WSD) Challenge.

## Data

The data must be saved in the `./data/` folder. The data is available here: https://raganato.github.io/vwsd/. The datasets for train and trial can be simply downloaded and extracted. For the test data, we chose to use the resized dataset. Therefore, the extracted test images must be saved to `./data/test_images_resized/`, while the gold labels directly reside in the `./data/` folder. 

## Baseline

Our two baseline models can be found in the directory with the same name. We wrote our own version of the CLIP baseline used in the paper. Additionally, we implemented an improved version using the more performant SigLIP model.

## Models

To solve the challenge, we created multiple custom models. We are using a CLIP-like approach and an ALIGN-like approach. Each model consists of a pretrained Visionencoder and a pretrained Textencoder.

Visionencoders:
- ResNet-50
- ViT
- EfficientNetB0
- ConvNeXt
- ConvNeXt V2

Textencoders:
- DistillBert
- GPT2

A simple variant to test is in `model.py`. This model uses ResNet-50 together with DistillBERT. It can be trained with `train.py` and tested with `test.py`.

To maintain a better overview, the models are grouped into four files. All Files have a similar structure. The file `clip_models.py` contains all models using the CLIP-like approach. There is a custom encoder model for each Vision- and Textencoder customizing them to our needs. An encoder model downloads the corresponding pretrained weights for the Encoder. A Feedforward Net is used for the projection. As the Input size varies for the different Encoders, this number is derived from the downloaded pretrained models. The custom encoder models are put together to form a custom CLIP-like model. Additionally, to forward and loss function, each model also provides a functio named top_images. This function takes a batch of images and a text. It outputs the similarities between each image and the text.

The file `align_models.py` contains the same model combinations as in the previous file. However, as this file contains the ALIGN-like models, the custom encoder models for the Visionencoders does not use the projection. Only the Textencoders use a projection. The output dimension of the projection is customized to the output dimension of the used Visionencoder.

Likewise the files `clip_models_glove.py` and `align_models_glove.py` contain another set of models. These models use GloVe to extend the context by creating synonyms for each word in the text. Therefore, the files contain additional functions `extend_text_with_synonyms` which is used in training and `extend_text_with_synonyms_inference` which are used in the inference phase.

The models can be trained with train_alternative_models. Here you can specify which models you want to train. You can also further train one of the model checkpoints, which are stored in the `results` folder. The models can be testet with `test.py`. It will return the results for each test datapoint and the final values for HIT@1 and MRR. 

## Dataset

We wrote two jupyter notebooks to familiarize ourselves with the provided data.

- data-analysis.ipynb: This notebook gives a small glimpse into the training and test data.
- data-loader.ipynb: This notebook was primarily used to develop our pytorch dataset class. It also contains example outputs of said class and demonstrates all features of the implemented dataset.

The finished dataset class resided in the dataset.py file. This file then gets imported into our model classes for loading the training or test data. 

The dataset can return training, test or validation data. It is possible to select the language of the returned set and set image as well as text transformations to be applied to the images. There is also an option to enable the data augmentation used by us. Additionally, a translated version of the farsi and italian versions can be selected.
The file also contains a second dataset class used for our work with the Flicker30K dataset.
