# ImageSimilarity
Techniques/Models for Image similarity search


1. Dino V2 with Faiss DB
2. Resnet Feature Extraction

Steps to run Dino V2:

1. Create python environment using requirements.txt
2. Run [processDino.py](processDino.py) to create embeddings and store them in a vector DB for images. (You may skip this step if you already have embeddings)
3. Run [rundino.py](rundino.py) to search for similar images. It will ask for absolute path of the image you want to search for.

Steps to run Resnet Feature Extraction:
1. Create python environment using requirements.txt
2. Run [processResnet.py](processResnet.py) to create embeddings and store them in a vector DB for images. (You may skip this step if you already have embeddings)
3. Run [runresnet.py](runresnet.py) to search for similar images. It will ask for absolute path of the image you want to search for.

Note : You can change any hyperparameters in the code as per your requirement from configs.
