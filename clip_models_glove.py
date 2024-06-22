import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.models import efficientnet_b0
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers import ViTModel, GPT2Model, GPT2Tokenizer

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import timm
import gensim.downloader as api

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"#
#import train


BATCH_SIZE = 128
EMBED_DIM = 512
TRANSFORMER_EMBED = 768
IMAGE_SIZE = 224

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
    

class VisionEncoderRes(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        base = models.resnet50(pretrained=True)
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

class VisionEncoderViT(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        d_in = self.base.config.hidden_size
        self.projection = Projection(d_in, d_out)
        
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        outputs = self.base(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        projected_vec = self.projection(cls_token)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

class VisionEncoderEfficient(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = efficientnet_b0(pretrained=True)
        d_in = self.base.classifier[1].in_features
        self.base.classifier = nn.Identity()

        self.projection = Projection(d_in, d_out)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.base(x)
        projected_vec = self.projection(features)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

class VisionEncoderConvNext(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = timm.create_model('convnext_base', pretrained=True)
        d_in = self.base.head.fc.in_features
        self.base.head.fc = nn.Identity()
        self.projection = Projection(d_in, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.base(x)
        projected_vec = self.projection(features)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len
    
class VisionEncoderConvNextV2(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = timm.create_model('convnextv2_base', pretrained=True)
        d_in = self.base.head.fc.in_features
        self.base.head.fc = nn.Identity()
        self.projection = Projection(d_in, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.base(x)
        projected_vec = self.projection(features)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len
    
class TextEncoderBERT(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        self.projection = Projection(TRANSFORMER_EMBED, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids, attention_mask=attention_mask)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len
    
class TokenizerBERT:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, truncation=True, padding=True, return_tensors="pt"
        )
    
class TextEncoderGPT2(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = GPT2Model.from_pretrained("gpt2")
        d_in = self.base.config.hidden_size

        self.projection = Projection(d_in, d_out)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids, attention_mask=attention_mask)[0]  # pass only input_ids
        out = out[:, -1, :]  # get last token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len
    
class TokenizerGPT2:
    def __init__(self, tokenizer: GPT2Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, x: str) -> GPT2Tokenizer:
        return self.tokenizer(
            x, truncation=True, padding=True, return_tensors="pt"
        )
    
class CustomModelResBERT(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderRes(EMBED_DIM)
        self.caption_encoder = TextEncoderBERT(EMBED_DIM)
        self.tokenizer = TokenizerBERT(AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelResBERT"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"], text["attention_mask"])

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities

class CustomModelViTBERT(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderViT(EMBED_DIM)
        self.caption_encoder = TextEncoderBERT(EMBED_DIM)
        self.tokenizer = TokenizerBERT(AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelViTBERT"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"], text["attention_mask"])

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities
    
class CustomModelEfficientBERT(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderEfficient(EMBED_DIM)
        self.caption_encoder = TextEncoderBERT(EMBED_DIM)
        self.tokenizer = TokenizerBERT(AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelEfficientBERT"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"], text["attention_mask"])

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities

class CustomModelConvNextBERT(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderConvNext(EMBED_DIM)
        self.caption_encoder = TextEncoderBERT(EMBED_DIM)
        self.tokenizer = TokenizerBERT(AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelConvNextBERT"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"], text["attention_mask"])

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities
    
class CustomModelConvNextV2BERT(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderViT(EMBED_DIM)
        self.caption_encoder = TextEncoderBERT(EMBED_DIM)
        self.tokenizer = TokenizerBERT(AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelConvNextV2BERT"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"], text["attention_mask"])

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities

class CustomModelResGPT2(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderRes(EMBED_DIM)
        self.caption_encoder = TextEncoderGPT2(EMBED_DIM)
        self.tokenizer = TokenizerGPT2(AutoTokenizer.from_pretrained("gpt2"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelResGPT2"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        text_input_ids = text["input_ids"].squeeze(1).to(self.device)  # Ensure correct shape
        attention_mask = text["attention_mask"].squeeze(1).to(self.device)  # Ensure correct shape

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text_input_ids, attention_mask)

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities

class CustomModelViTGPT2(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderViT(EMBED_DIM)
        self.caption_encoder = TextEncoderGPT2(EMBED_DIM)
        self.tokenizer = TokenizerGPT2(AutoTokenizer.from_pretrained("gpt2"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelViTGPT2"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        text_input_ids = text["input_ids"].squeeze(1).to(self.device)  # Ensure correct shape
        attention_mask = text["attention_mask"].squeeze(1).to(self.device)  # Ensure correct shape

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text_input_ids, attention_mask)

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities
    
class CustomModelEfficientGPT2(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderEfficient(EMBED_DIM)
        self.caption_encoder = TextEncoderGPT2(EMBED_DIM)
        self.tokenizer = TokenizerGPT2(AutoTokenizer.from_pretrained("gpt2"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelEfficientGPT2"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        text_input_ids = text["input_ids"].squeeze(1).to(self.device)  # Ensure correct shape
        attention_mask = text["attention_mask"].squeeze(1).to(self.device)  # Ensure correct shape

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text_input_ids, attention_mask)

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

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

class CustomModelConvNextGPT2(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderConvNext(EMBED_DIM)
        self.caption_encoder = TextEncoderBERT(EMBED_DIM)
        self.tokenizer = TokenizerGPT2(AutoTokenizer.from_pretrained("gpt2"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelConvNextGPT2"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        text_input_ids = text["input_ids"].squeeze(1).to(self.device)  # Ensure correct shape
        attention_mask = text["attention_mask"].squeeze(1).to(self.device)  # Ensure correct shape

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text_input_ids, attention_mask)

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities
    
class CustomModelConvNextV2GPT2(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderViT(EMBED_DIM)
        self.caption_encoder = TextEncoderBERT(EMBED_DIM)
        self.tokenizer = TokenizerGPT2(AutoTokenizer.from_pretrained("gpt2"))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "CustomModelConvNextV2GPT2"
        self.glove_model = api.load("glove-wiki-gigaword-50")

    def forward(self, images, text):
        text = extend_text_with_synonyms(text, self.glove_model)
        text = self.tokenizer(text)
        text = {key: tensor.to(self.device) for key, tensor in text.items()}

        text_input_ids = text["input_ids"].squeeze(1).to(self.device)  # Ensure correct shape
        attention_mask = text["attention_mask"].squeeze(1).to(self.device)  # Ensure correct shape

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text_input_ids, attention_mask)

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        caption_embed = F.normalize(caption_embed, p=2, dim=-1)

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
        extended_texts = extend_text_with_synonyms_inference(text, self.glove_model)

        tokenized_texts = self.tokenizer(extended_texts)
        tokenized_texts = {key: tensor.to(self.device) for key, tensor in tokenized_texts.items()}

        caption_embed = self.caption_encoder(tokenized_texts["input_ids"], tokenized_texts["attention_mask"])

        similarities = []

        for image in images:
            image_embed = self.vision_encoder(image.to(self.device))
            similarities.append(F.cosine_similarity(image_embed, caption_embed, dim=1).item())

        #top_image = np.argsort(similarities)[-1:][::-1]

        return similarities
    
def get_synonyms(word, model, top_n=3):
    if word in model:
        return [synonym for synonym, _ in model.most_similar(word, topn=top_n)]
    else:
        return []

def extend_text_with_synonyms(texts, model, top_n=3):
    # Tokenize texts
    tokenized_texts = [text.split() for text in texts]
    extended_texts = []

    for tokens in tokenized_texts:
        extended_tokens = []
        
        # Find synonyms for each token
        for token in tokens:
            synonyms = get_synonyms(token, model, top_n)
            extended_tokens.append([token] + synonyms)  # Include original token and synonyms

        extended_texts.append(extended_tokens)
    
    # Convert extended_texts to list of strings
    extended_texts = [' '.join([item for sublist in tokens for item in sublist]) for tokens in extended_texts]
    
    return extended_texts

def extend_text_with_synonyms_inference(text, model, top_n=3):
    words = text.split()
    extended_text = []
    
    for word in words:
        extended_text.append(word)
        synonyms = get_synonyms(word, model, top_n)
        extended_text.extend(synonyms)
    
    return " ".join(extended_text)