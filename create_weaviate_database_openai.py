import os
import time
import argparse
import pickle

from tqdm import tqdm
import weaviate

WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "ScreenshotQueryOPENAI"

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

    
    print("Getting features from .pkl files and insert in the database..."); start = time.time()
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image in tqdm(images):
        desc_path = f"{os.path.join(image_dir, image)}.txt"
        if os.path.exists(desc_path):
            with open(desc_path, "r") as f:
                description = f.read()

            # read pickle file of the embedding
            embed_path = f"{desc_path.replace('.txt', '')}_embed.pkl"
            if os.path.exists(embed_path):
                with open(embed_path, "rb") as f:
                    feat = pickle.load(f)
            else:
                print("WARNING: Embedding file not found for", image)

            # Encode the description    
            client.data_object.create({
                "filename": image,
                "description": description
            }, CLASS_NAME, vector=feat)

        else:
            print("WARNING: Description file not found for", image)

    print("Database populated in", time.time() - start, "seconds")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--image_dir", type=str, required=True)
    args = argparse.parse_args()
    create_weaviate_database(args.image_dir)