from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as tr
import pandas as pd
import torch
import warnings

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", message="Corrupt EXIF data.  Expecting to read 2 bytes but only got 0.")

IMAGE_SIZE = 256
BATCH_SIZE = 1

class VisualWSDDataset(Dataset):
    def __init__(self, mode="train", image_transform=None, text_transform=None, tokenizer=None, test_lang='en', translate=False, augmentation=False):
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.tokenizer = tokenizer
        self.mode = mode
        self.translate = translate

        self.base_path = './data/semeval-2023-task-1-V-WSD-train-v1/train_v1/'
        self.data_txt_path = self.base_path + 'train.data.v1.txt'
        self.gold_txt_path = self.base_path + 'train.gold.v1.txt'
        self.image_path = self.base_path + '/train_images_v1/'

        if mode == "test":
            self.base_path = './data/'
            self.data_txt_path = self.base_path + test_lang + '.test.data.v1.1.txt'
            self.gold_txt_path = self.base_path + test_lang + '.test.gold.v1.1.txt'
            self.image_path = self.base_path + '/test_images_resized/'
            self.text_translation = './it_en.txt'

            if test_lang == 'fa':
                self.data_txt_path = self.base_path + test_lang + '.test.data.txt'
                self.gold_txt_path = self.base_path + test_lang + '.test.gold.txt'
                self.text_translation = './fa_en.txt'

            # open translation file
            trans_file = open(self.text_translation, 'r')
            self.gold_translation = trans_file.readlines()

        elif mode == "val":
            self.base_path = './data/semeval-2023-task-1-V-WSD-train-v1/trial_v1/'
            self.data_txt_path = self.base_path + 'trial.data.v1.txt'
            self.gold_txt_path = self.base_path + 'trial.gold.v1.txt'
            self.image_path = self.base_path + '/trial_images_v1/'
        

        # load txts
        self.data_df = pd.read_csv(self.data_txt_path, delimiter = "\t", header=None)
        self.gold_df = pd.read_csv(self.gold_txt_path, delimiter = "\t", header=None)
        self.data_gold_df  = pd.concat([self.data_df.iloc[:, 0], self.data_df.iloc[:, 1], self.gold_df.iloc[:, 0]], axis=1, keys=['label', 'label_context', 'img_name'])
        
        if self.tokenizer != None:
            self.gold_token = self.tokenizer(self.data_gold_df['label_context'].to_list())

        self.augmentation = augmentation
        self.augmentations = [
            tr.RandomHorizontalFlip(p=1),
            tr.ColorJitter(brightness=0.5),
            tr.RandomRotation(30)
            # other augmentations possible: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
        ]

    def __len__(self):
        if self.mode == "test" or not self.augmentation:
            return len(self.data_gold_df)
        else:
            return len(self.data_gold_df) * (len(self.augmentations) + 1)

    def __getitem__(self, idx):
        if self.augmentation:
            original_idx = idx // (len(self.augmentations) + 1)
        else:
            original_idx = idx
        
        augment_idx = idx % (len(self.augmentations) + 1)
        label = self.data_gold_df.iloc[original_idx]['label']
        label_context = self.data_gold_df.iloc[original_idx]['label_context']
        correct_image_name = self.data_gold_df.iloc[original_idx]['img_name']
        images_series = self.data_df.iloc[original_idx][2:]
        correct_image_idx = images_series[images_series == correct_image_name].index[0]-2
        images = []

        # swap label contex with translated one
        if self.translate:
            label_context = self.gold_translation[original_idx]

        if self.mode == "test":
            for item in images_series:
                    images.append(Image.open(self.image_path + item).convert('RGB'))
            correct_image = images[correct_image_idx]
        else:
            correct_image = Image.open(self.image_path + images_series[correct_image_idx+2]).convert('RGB')
            if augment_idx > 0 and self.augmentation:
                augment = self.augmentations[augment_idx - 1]
                correct_image = augment(correct_image)

        if self.image_transform:
            correct_image = self.image_transform(correct_image)
            for idx in range(len(images)):
                images[idx] = self.image_transform(images[idx])

        if self.text_transform:
            label = self.text_transform(label)
            label_context = self.text_transform(label_context)

        if self.tokenizer != None:
            # labels are the correct images
            # input_ids and attention_mask are tokenized text
            item = {key: torch.tensor(val[original_idx]) for key, val in self.gold_token.items()}
            item['images'] = torch.tensor(correct_image).detach()
            return item

        return {'label': label, 'label_context': label_context, 'correct_idx': correct_image_idx, 'correct_img': correct_image, 'imgs': images}

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import torch
from zipfile import ZipFile
import urllib.request
import json
import collections
from datasets import load_dataset

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.cap_per_image = 2

    def __len__(self):
        return self.dataset.num_rows["test"] * self.cap_per_image

    def __getitem__(self, idx):
        original_idx = idx // self.cap_per_image
        # image_path = self.dataset[idx]["image_path"]
        image = self.dataset["test"][original_idx]["image"].convert("RGB")
        image = self.transform(image)

        # You might need to adjust the labels based on your task
        caption = self.dataset["test"][original_idx]["caption"][idx % self.cap_per_image]

        return {'label': caption, 'label_context': caption, 'correct_idx': 0, 'correct_img': image, 'imgs': []}