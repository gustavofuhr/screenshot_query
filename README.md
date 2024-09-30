## 🔍 screenshot_query - Natural language image retrieval

Make natural language queries to retrieve relevant screenshots. Check the post that I wrote explaining this
project at my website: [gfuhr.me](https://gfuhr.me/) FIX LINK

This is basically a collection of scripts and notebooks to achieve this:

GIF

#### How to use:

1. Create image descriptions using the `generate_image_descriptions_openai.py`. Just provide the `--image-dir` parameter. This will generate in the image directory text descriptions of the image contents, one per file, named something like "image01.png.txt". It should take some time, depending on the number of images you have. When rerun, the script will resume the processing.

2. Start a weaviate database, which will be used to store description embeddings.

```
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.3
```

os use the `start_weviate.sh`.

3. (Optional). If you want to use OpenAI API to create the description embeddings, you first need
to run `create_descriptions_openai_embeddings.py`. This should take a couple of minutes and will create pickle
files in the provided directory. If you choose to use SBERT embeddings, you can skip this step, since they will
be create in the next one.

4. Populate the weaviate base, using SBERT or OPEANAI embeddings: `create_weaviate_database_openai.py` or
`create_weaviate_database_sbert.py`.

5. Interact with your images using the notebook `query_screenshot_openai_bigprompt.ipynb`. You should set the
embedding you want to use at the beginning of the code.

Have fun.

