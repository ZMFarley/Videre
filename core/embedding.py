from PIL import Image
import requests
from transformers import AutoImageProcessor, SiglipVisionModel
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import time
from pathlib import Path
#Loading Pretrained/Frozen embedders
model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast = True)

#Load in Relevant GPU or CPU to run model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
#load in relevant datasets

'''
This is the saved code used to load in my corresponding datasets. They were frozen in comments for later refactoring while i 
Began iterating the progress on my overarching project 
#Obtained from https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney
midjourney_ai_images = load_dataset("ideepankarsharma2003/AIGeneratedImages_Midjourney", split ="train", streaming = True)
midjourney_list = []
batch = []

for index, row in enumerate(midjourney_ai_images):
    if (len(batch) < 128 and index != 1000):
        batch.append(row["image"])
    if (index == 1000):
        break
    else:
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Convert the tensor to a numpy array for storage 
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        midjourney_list.append(embeddings)
        batch.clear()
#Save the embeddings for temporary use
np.save("embeddings_midjourney", embeddings)

coco_images = Path(r"C:\Users\zacha\projects\CIS430\Videre\core\val2017").glob('*.jpg')
coco_list = []
for index, image in enumerate(coco_images):
    if(index < 1000):
        coco_list.append(image)
    else:
        break
batch = []
coco_embedding_list = []
for index, row in enumerate(coco_list):
    if (len(batch) < 128):
        batch.append(Image.open(coco_list[index]).convert("RGB"))
    else:
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Convert the tensor to a numpy array for storage 
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        coco_embedding_list.append(embeddings)
        batch.clear()
#Save the embeddings for temporary use
np.save("embeddings_coco", embeddings)

'''

#for i in range (1000):
clf = LogisticRegression()

#Helper function to load in model 
def __load_model():
    #Loading Pretrained/Frozen embedders
    model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast = True)
    return model, processor

#Function to convert user's inputted image
def convert_input_image(input):
    #Load in embedder
    model, processor = __load_model()

    #Open image
    image = Image.open(input)
    #Disables gradient calculation as we are only embedding vectors, no need for model training
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

    #Convert the tensor to a numpy array for storage 
    embeddings = outputs.pooler_output.detach().cpu().numpy()
    return embeddings