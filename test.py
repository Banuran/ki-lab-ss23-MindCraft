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

BATCH_SIZE = 16
EMBED_DIM = 512
TRANSFORMER_EMBED = 768
IMAGE_SIZE = 255

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds
    

class VisionEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        base = models.resnet34(pretrained=True)
        d_in = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.projection = Projection(d_in, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len
    
class TextEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        self.projection = Projection(TRANSFORMER_EMBED, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len
    
class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, truncation=True, padding=True, return_tensors="pt"
        )
    
class CustomModel(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(EMBED_DIM)
        self.caption_encoder = TextEncoder(EMBED_DIM)
        self.tokenizer = Tokenizer(AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, images, text):
        text = self.tokenizer(text).to(self.device)

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"])

        similarity = caption_embed @ image_embed.T

        loss = self.CLIP_loss(similarity)
        img_acc, cap_acc = metrics(similarity)

        return loss, img_acc, cap_acc
    
    def CLIP_loss(self, logits: torch.Tensor) -> torch.Tensor:
        n = logits.shape[1]      # number of samples
        labels = torch.arange(n).to(self.device) # Create labels tensor
        # Calculate cross entropy losses along axis 0 and 1
        loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
        loss_t = F.cross_entropy(logits, labels, reduction="mean")
        # Calculate the final loss
        loss = (loss_i + loss_t) / 2

        return loss
    
    def top_image(self, images, text):
        text = self.tokenizer(text).to(self.device)
        caption_embed = self.caption_encoder(text["input_ids"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities
    
def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


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

    eval_model = CustomModel().to(device)
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