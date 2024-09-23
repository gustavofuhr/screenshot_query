import os
import time
import argparse

from tqdm import tqdm
import weaviate
from sentence_transformers import SentenceTransformer

WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "ScreenshotQuerySBERT"
SBERT_MODEL = "all-MiniLM-L6-v2"

def create_weaviate_database(image_dir):
    # Connect to a Weaviate instance
    client = weaviate.Client(WEAVIATE_URL)

    if any(cls['class'] == CLASS_NAME for cls in client.schema.get()['classes']):
        print(f"Class {CLASS_NAME} already exists. Do you want to erase it?")
        erase = input(f"Erase class {CLASS_NAME}? (y/n): ")
        if erase == "y":
            client.schema.delete_class(CLASS_NAME)
        else:
            print("Exiting...")
            exit()

    print("Creating schema in weaviate..."); start = time.time()
    # Define the schema
    class_obj = {
        "class": CLASS_NAME,
        "properties": [
            {"name": "filename", "dataType": ["string"]},
            {"name": "description", "dataType": ["string"]}
        ]
    }
    client.schema.create_class(class_obj)
    print("Schema created in", time.time() - start, "seconds")

    print(f"Instanciating SentenceTransformer model {SBERT_MODEL}..."); start = time.time()
    model = SentenceTransformer(SBERT_MODEL)
    print("Model instanciated in", time.time() - start, "seconds")


    print("Getting features and populating database..."); start = time.time()
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image in tqdm(images):
        image_path = os.path.join(image_dir, image)
        if os.path.exists(f"{image_path}.txt"):
            with open(f"{image_path}.txt", "r") as f:
                description = f.read()

            # Encode the description    
            feat = model.encode(description)
            client.data_object.create({
                "filename": image,
                "description": description
            }, CLASS_NAME, vector=feat.tolist())
    print("Database populated in", time.time() - start, "seconds")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--image_dir", type=str, required=True)
    args = argparse.parse_args()
    create_weaviate_database(args.image_dir)