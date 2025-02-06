import faiss
import numpy as np
import os

class FaissSearcher:
    def __init__(self, index_path, names_path):
        if not os.path.exists(index_path) or not os.path.exists(names_path):
            raise FileNotFoundError("FAISS index or image names file not found. Please generate them first.")
        
        self.index = faiss.read_index(index_path)
        self.image_names = np.load(names_path)

    def search_sim(self, query_embedding, top_k=5):
        distances, indices = self.index.search(query_embedding.reshape(1,-1), top_k)
        similar_images = [self.image_names[i].item() for i in indices[0] if i < len(self.image_names)]
        return similar_images, distances[0]
