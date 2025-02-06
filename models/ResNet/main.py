#!/usr/bin/env python3
from PIL import Image
from models.ResNet.similarity import FaissSearcher
from models.ResNet.generateEmbeddings import ImageEmbedding
import yaml

def main(query, config):
    embedder = ImageEmbedding()
    embedder.model.eval()
    searcher = FaissSearcher(index_path=config['db']['indexPath'], names_path= config['data']['imageNames'])

    embedding = embedder.get_embedding(query)
    results, distances = searcher.search_sim(embedding.reshape(-1), config['hyperParameters']['topk'])
    print(f"Query Image: {query}\n")
    Image.open(f"{query}").show()

    print(f"Similar Images: {results}")
    for i, image_id in enumerate(results):
        Image.open(f"{config['data']['dataPath']}/{image_id}.jpg").show()
        print(f"Similarity: {distances[i]}")

if __name__ == "__main__":
    with open('configs/resnet50.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("Please type the location of .jpg Image")
    query = input()  ### image path
    main(query, config)
