import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from torchvision import transforms as tt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from datetime import datetime
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import dataset as wsd
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import model as md
import model_disk_helper as mdh

BATCH_SIZE = 64
IMAGE_SIZE = 224


def main():

    scale = tt.Resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = tt.ToTensor()
    image_composed = tt.transforms.Compose([scale, tensor])

    train_set = wsd.VisualWSDDataset(mode="train", image_transform=image_composed)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = md.CustomModel().to(device)

    print(model.device)
    
    optimizer = torch.optim.Adam([
        {'params': model.vision_encoder.parameters()},
        {'params': model.caption_encoder.parameters()}
    ]   , lr=model.lr)

    ## train loop
    start_epoch = 0
    num_epochs = 3

    batch_zero = True
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch in train_loader:
            image = batch["correct_img"].to(device)
            text = batch["label_context"]
            # images, text = batch
            loss, img_acc, cap_acc = model(image, text)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_zero:
                print(f"Epoch [{0}/{num_epochs}], Batch Loss: {loss.item()}")
                batch_zero = False


        # Print training statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")

    print("Training complete.")
    ## end train loop

    path = mdh.save_model(model, True)
    print("saved model to: " + path)

main()
    