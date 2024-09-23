import os
import argparse
import pickle

import requests
import tqdm


def get_openai_embedding(text: str, model: str = "text-embedding-3-small"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    payload = {
        "model": model,
        "input": text
    }

    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload).json()
    if "error" in response:
        raise Exception(f"Error: {response}")
    
    return response["data"][0]["embedding"]

def create_descriptions_openai_embeddings(image_dir):
    image_descriptions = [f"{f}.txt" for f in os.listdir(image_dir) \
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for desc in tqdm.tqdm(image_descriptions):
        desc_path = os.path.join(image_dir, desc)
        embed_path = f"{desc_path.replace('.txt', '')}_embed.pkl"
        if not os.path.exists(embed_path):
            with open(desc_path, "r") as f:
                desc = f.read()
            
            embedding = get_openai_embedding(desc)
            with open(embed_path, "wb") as f:
                pickle.dump(embedding, f)
            
    print("Done")

    


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_dir", type=str, required=True,
                       help="the image dir where the descriptions are as text files",)
    
    args = parse.parse_args()
    create_descriptions_openai_embeddings(args.image_dir)