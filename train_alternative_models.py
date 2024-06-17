import torch
from torchvision import transforms as tt
from torch.utils.data import DataLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

import dataset as wsd
import alternative_models as md
import model_disk_helper as mdh

BATCH_SIZE = 64
IMAGE_SIZE = 224

NUM_EPOCHS = 10

def main():

    scale = tt.Resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = tt.ToTensor()
    image_composed = tt.transforms.Compose([scale, tensor])

    train_set = wsd.VisualWSDDataset(mode="train", image_transform=image_composed)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = [md.CustomModelResBERT, md.CustomModelViTBERT, md.CustomModelEfficientBERT,
              md.CustomModelResGPT2, md.CustomModelViTGPT2, md.CustomModelEfficientGPT2]

    for model_variant in models:
        start_time = time.time()

        model = model_variant().to(device)

        print(model.device)
        
        optimizer = torch.optim.Adam([
            {'params': model.vision_encoder.parameters()},
            {'params': model.caption_encoder.parameters()}
        ]   , lr=model.lr)

        batch_zero = True
        for epoch in range(0, NUM_EPOCHS):
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
                    print(f"Epoch [{0}/{NUM_EPOCHS}], Batch Loss: {loss.item()}")
                    batch_zero = False


            # Print training statistics
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch Loss: {loss.item()}")

        elapsed_time = time.time() - start_time
        print("Training complete.")
        ## end train loop

        path = mdh.save_model(model, NUM_EPOCHS, elapsed_time)
        print("saved model to: " + path)

        print(model.name)

main()
    