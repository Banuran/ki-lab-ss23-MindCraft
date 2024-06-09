from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as tt
import pandas as pd
import numpy as np

IMAGE_SIZE = 256
BATCH_SIZE = 1

class VisualWSDDataset(Dataset):
    def __init__(self, train, image_transform=None, text_transform=None, test_lang='en'):
        self.train = train
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.base_path = './data/semeval-2023-task-1-V-WSD-train-v1/train_v1/'
        self.data_txt_path = self.base_path + 'train.data.v1.txt'
        self.gold_txt_path = self.base_path + 'train.gold.v1.txt'
        self.image_path = self.base_path + '/train_images_v1/'

        if not train:
            self.base_path = './data/'
            self.data_txt_path = self.base_path + test_lang + '.test.data.v1.1.txt'
            self.gold_txt_path = self.base_path + test_lang + '.test.gold.v1.1.txt'
            self.image_path = self.base_path + '/test_images_resized/'

            if test_lang == 'fa':
                self.data_txt_path = self.base_path + test_lang + '.test.data.txt'
                self.gold_txt_path = self.base_path + test_lang + '.test.gold.txt'
        

        # load txts
        self.data_df = pd.read_csv(self.data_txt_path, delimiter = "\t", header=None)
        self.gold_df = pd.read_csv(self.gold_txt_path, delimiter = "\t", header=None)
        self.data_gold_df  = pd.concat([self.data_df.iloc[:, 0], self.data_df.iloc[:, 1], self.gold_df.iloc[:, 0]], axis=1, keys=['label', 'label_context', 'img_name'])

    def __len__(self):
        return len(self.data_gold_df)

    def __getitem__(self, idx):
        label = self.data_gold_df.iloc[idx]['label']
        label_context = self.data_gold_df.iloc[idx]['label_context']
        correct_image_name = self.data_gold_df.iloc[idx]['img_name']
        images_series = self.data_df.iloc[idx][2:]
        correct_image_idx = images_series[images_series == correct_image_name].index[0]-2
        images = []
        for item in images_series:
                images.append(Image.open(self.image_path + item).convert('RGB'))
        correct_image = images[correct_image_idx]

        if self.image_transform:
            correct_image = self.image_transform(correct_image)
            for idx in range(len(images)):
                images[idx] = self.image_transform(images[idx])

        if self.text_transform:
            label = self.text_transform(label)
            label_context = self.text_transform(label_context)


        return {'label': label, 'label_context': label_context, 'correct_idx': correct_image_idx, 'correct_img': correct_image, 'imgs': images}
