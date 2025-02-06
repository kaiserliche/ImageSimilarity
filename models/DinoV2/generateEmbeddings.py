#!/usr/bin/env python3
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import random
import yaml
import os
class ImageEmbedding:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def get_embedding(self, img):
        image = Image.open(img).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            return self.model(image)

if __name__ == "__main__":
    print(os.getcwd())
    with open('configs/dinov2.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataPath = config['data']['dataPath']
    
    embedder = ImageEmbedding()
    df = pd.read_csv(config['data']['processedPath'], header=None)
    rows = df[0].values.tolist()
    random.shuffle(rows)

    sample = config['data']['sample']
    paths = [f"{dataPath}{str(index)}.jpg" for index in rows[:sample]]
    

    embeddings = torch.cat([embedder.get_embedding(img) for img in paths], dim=0)
    embdict = {str(ind): emb for ind, emb in zip(rows[:sample], embeddings)}
    torch.save(embdict, config['data']['embeddingPath'])
    print("Embeddings generated successfully")
