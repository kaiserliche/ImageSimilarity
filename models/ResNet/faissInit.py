#!/usr/bin/env python3
import numpy as np
import torch
import faiss
import yaml

def main():
    with open('configs/resnet50.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    emb_dict = torch.load(config['data']['embeddingPath'], map_location="cpu")
    image_names = list(emb_dict.keys())
    embeddings = np.stack([emb_dict[img].numpy() for img in image_names]).astype('float32')

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)  #### can use HNSW for faster retrieval
    index.train(embeddings)
    index.add(embeddings)

    faiss.write_index(index, config['db']['indexPath'])
    np.save(config['data']['imageNames'], np.array(image_names))  # Faster than JSON
    print("FAISS index and image names saved successfully")

if __name__ == "__main__":
    main()
