from PIL import Image
import requests
from transformers import AutoImageProcessor, SiglipVisionModel
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
import time
from pathlib import Path
import os 
import joblib
from itertools import islice
from io import BytesIO

#Class to hold embedding model, and relevant helper functions
class Embedder: 
    def __init__(self) -> None:
        self.model, self.processor, self.device = self.__load_model()

    #Helper function to load in model 
    def __load_model(self) -> tuple[SiglipVisionModel, AutoImageProcessor, str]:
        try:
            #Loading Pretrained/Frozen embedders
            model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
            processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast = True)
        except Exception as e:
            raise RuntimeError("Failure to load model or processor") from e 
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            #Load in Relevant GPU or CPU to run model
            model.to(device)
        except RuntimeError as e:
            if device == 'cuda':
                print("Failure to load CUDA for GPU, switching to CPU")
                device = "cpu"
                model.to(device)
            else:
                raise
        return model, processor, device 
    
    def _embed(self, batch: list[Image.Image]) -> np.ndarray:
        if not batch:
            raise ValueError("Recieved empty batch, unable to embed")
        if not all(isinstance(image, Image.Image)for image in batch):
            raise TypeError("Batch must contain only PIL.Image.Image objects")
    #Convert images into embeddings
        with torch.no_grad():
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            return embeddings

def embed_huggingface_datasets(embedder: Embedder, dataset: str, training_split: str, save_file_name: str, num_images: int, batch_size: int) -> np.ndarray:
    images_list = load_dataset(dataset, split = training_split, streaming = True)
    embedding_list = []
    batch = []

    if num_images <= 0:
        raise ValueError("Number of images to embed must be > 0")
    if batch_size <= 0:
        raise ValueError("Batch size to embed must be > 0")
    
    #Batch images into specificed length groups up until the last index
    for index, row in enumerate(images_list):
        if (index == num_images):
            break
        batch.append(row["image"].convert("RGB"))
        if (len(batch) == batch_size):
            embedding_list.append(embedder._embed(batch))
            batch.clear()
    if (len(batch) > 0):
        embedding_list.append(embedder._embed(batch))
        batch.clear()
    #Save the embeddings for temporary use
    embedding_matrix = np.concatenate(embedding_list, axis = 0)
    np.save(save_file_name, embedding_matrix)
    return embedding_matrix 

def embed_locally_saved_datasets(embedder: Embedder, dataset_path: str, save_file_name: str, num_images: int, batch_size: int, index_start: int, index_end: int) -> np.ndarray:
    #Pull images from local path (Requires adjustment for given users machine) and store them locally 
    images = Path(dataset_path).glob('*.jpg')
    images_list = []
    batch = []
    embedding_list = []

    #Section to verify input values
    if num_images <= 0:
        raise ValueError("Number of images to embed must be > 0")
    if batch_size <= 0:
        raise ValueError("Batch size to embed must be > 0")
    if index_start < 0:
        raise ValueError("Starting index must be >= 0")
    if index_end is not None and index_end < index_start:
        raise ValueError("Starting index must be > index_start")
    
    #grab first 500 images after training data set of 1000
    for index, image in enumerate(islice(images, index_start, index_end)):
        if(index < num_images):
            images_list.append(image)
        else:
            break
    for row in images_list:
        #batch images into groups of 128 for embedding
        try:
            batch.append(Image.open(row).convert("RGB"))
        except Exception as e:
            print(f"Unable to read image path, skipping {row}: {e}")
            continue
        if (len(batch) == batch_size):
            #embed images
            embedding_list.append(embedder._embed(batch))
            batch.clear()
    if (len(batch) > 0):
        embedding_list.append(embedder._embed(batch))
        batch.clear()

    #Save the embeddings for temporary use
    if not embedding_list:
        raise RuntimeError("No embeddings generated for classifications")
    embedding_matrix = np.concatenate(embedding_list, axis = 0)
    np.save(save_file_name, embedding_matrix)
    return embedding_matrix

#Function to convert user's inputted image
def convert_input_image(input: bytes) -> np.ndarray:
    #Instiate embedding object to load relevant models
    embedder = Embedder()
    #Convert incoming image bytes to image and Open image
    bytes_to_image = BytesIO(input)
    image = [Image.open(bytes_to_image)]
    embedding = embedder._embed(image)
    return embedding

#Function for Full stack application to predict image 
def predict_image(input: bytes) -> dict:
    if not input:
        raise ValueError("Did not recieve image bytes")
    #Convert image to embeddings
    embeddings = convert_input_image(input)

    #Load classifer and predict the image's class and the probabilities of it
    try:
        clf = joblib.load("final_classifier.joblib")
    except Exception as e:
        raise RuntimeError(f"Unable to load classifier: {e}")
    
    predicted_class = clf.predict(embeddings)
    prob_real, prob_fake = clf.predict_proba(embeddings)[0]
    
    #Create a JSON payload for the API to return, and return the predicted values, typecasting them to regular floats instead of numpy floats
    results_payload = {"result": int(predicted_class[0]), "probability_real": float(prob_real), "probability_ai": float(prob_fake)}
    return results_payload
