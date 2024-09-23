import base64
import requests
import os
import argparse

import tqdm

# encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_description_openai(image_path):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What’s in this image?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    return response["choices"][0]["message"]["content"]

def generate_image_captions(image_dir):
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image in tqdm.tqdm(images):
        image_path = os.path.join(image_dir, image)
        if not os.path.exists(f"{image_path}.txt"):
            description = get_image_description_openai(image_path)
            with open(f"{image_path}.txt", "w") as f:
                f.write(description)

    print("Done")
    
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--image_dir", type=str, required=True)
    args = argparse.parse_args()
    generate_image_captions(args.image_dir)