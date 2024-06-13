import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import dataset as wsd
import numpy as np
import os
from torchvision import transforms as tt
os.environ["TOKENIZERS_PARALLELISM"] = "false"#
#import train
import sys

import model as md

IMAGE_SIZE = 255

# if k is 1 gives all instances with the correct prediction as top prediction
# if k > 1 the correct prediction is in the top k predictions of the model
def hit(results, k):
    counter = 0

    for r in results:
        sims = np.absolute(r[1])
        sorted = np.argsort(sims)[:k]

        if r[0] in sorted:
            counter += 1

    return counter / len(results)

def mrr(results):
    sum = 0

    for r in results:
        sims = np.absolute(r[1])
        sorted = np.argsort(sims)
        sum += 1/(np.where(sorted==r[0])[0][0]+1)

    return sum / len(results)

def test_loop(loader, model):
    results = []

    for i,batch in enumerate(loader):
        images = batch["imgs"]
        text = batch["label_context"]
        correct_idx = batch["correct_idx"].item()
        sims = model.top_image(images, text)

        results.append((correct_idx, sims))
        top_image = np.argsort(np.absolute(sims))[0]

        #print(np.argsort(np.absolute(sims)))
        print("batch: " + str(i+1) + "/" + str(len(loader)) + " predicted: " + str(top_image) + " correct: " + str(correct_idx))

        #if i == 50:
        #    break

    return results

def main():

    name = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scale = tt.Resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = tt.ToTensor()
    image_composed = tt.transforms.Compose([scale, tensor])

    eval_model = md.CustomModel().to(device)
    model_name = name
    eval_model.load_state_dict(torch.load("./results/" + model_name))
    eval_model.eval()
    print(eval_model.device)

    test_set = wsd.VisualWSDDataset(mode="test", image_transform=image_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)


    results = test_loop(test_loader, eval_model)

    print("hit@1: " + str(hit(results, 1)))
    print("mrr: " + str(mrr(results)))


main()