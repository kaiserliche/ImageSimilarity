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



Improvements :

For accuracy :

1. use of more advanced models used for this purpose like CLIP. 
2. Finetune the models for custom dataset. Dino V2 does not require much finetuning, but you can do it and depends on the complication of data. If data is too simple it may give you bad results. Finetuning on ResNet should improve embeddings/features.
3. Use of different similarity/distances. (cosine, euclidean, manhattan)
4. Ensembling the image embeddings with text given in metadata. It will improve accuracy not only by image but also by non visual features such as brands, season, etc.

For speed :
1. Use of quantized models. 
2. Lower dimensional embeddings.
3. Use of HNSW instead of flat index in vector database.
4. Use of more robust db's

