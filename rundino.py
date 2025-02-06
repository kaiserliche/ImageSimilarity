from models.DinoV2.main import main
import yaml

with open('configs/dinov2.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
print("Please type the location of .jpg Image")
query = input()  ### image path
main(query, config)