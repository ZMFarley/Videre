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
import os 
import joblib
from itertools import islice
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from io import BytesIO

#Class to hold embedding model, and relevant helper functions
class Embedder: 
    def __init__(self):
        self.model, self.processor, self.device = self.__load_model()
    #Helper function to load in model 
    def __load_model(self):
        #Loading Pretrained/Frozen embedders
        model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast = True)

        #Load in Relevant GPU or CPU to run model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        return model, processor, device 
    
    def _embed(self, batch):
    #Convert images into embeddings
        with torch.no_grad():
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            return embeddings

def create_embeddings_and_classifier():
    #load in relevant datasets and embedder
    embedder = Embedder()
    #Obtained from https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney
    midjourney_matrix = embed_huggingface_datasets(embedder, "ideepankarsharma2003/AIGeneratedImages_Midjourney", "train", "embeddings_midjourney", 1000, 128)

    #Images here were obtained from the COCO datast, val2017, using the first 1000 images
    coco_matrix = embed_locally_saved_datasets(embedder, r"C:\Users\zacha\projects\CIS430\Videre\core\val2017", "embeddings_coco", 1000, 128, 0, 1000)

    #natural images from https://www.kaggle.com/datasets/prasunroy/natural-images for real images, and 
    #https://www.kaggle.com/datasets/gpch2159/ai-vs-human-syn-imgs-v2-partial/data
    #for ai generated images using stable Diffusion XL
    natural_matrix = embed_locally_saved_datasets(embedder, r"C:\Users\zacha\projects\CIS430\Videre\core\natural_images\embedding_images", "embeddings_natural_images", 1000, 128, 0, 1000)
    ai_matrix = embed_locally_saved_datasets(embedder, r"C:\Users\zacha\projects\CIS430\Videre\core\stabilityai.stable-diffusion-xl-refiner-1.0_0.5_12_2025.02.25_05.15.08_846327", "embeddings_stableDiffXL", 1000, 128, 0, 1000)

    #create corresponding label arrays for each matrix 
    #then concatenate them into one label matrix that will correspond to the collapsed 
    midjourney_labels = [1] * 1000
    coco_labels =  [0] * 993
    natural_image_labels = [0] * 993
    stable_diff_labels = [1] * 993

    label_matrix = np.concatenate((midjourney_labels, coco_labels, natural_image_labels, stable_diff_labels), axis = 0)

    #Create the corresponding embeddings matrix to train the model 
    training_embedding_matrix = np.concatenate((midjourney_matrix, coco_matrix, natural_matrix, ai_matrix), axis = 0)

    #Create the logistic regression classifier, and train it on the relevant data, saving it afterwards 
    clf = LogisticRegression()

    clf.fit(training_embedding_matrix, label_matrix)

    #joblib.dump(clf, "test_classifier.joblib")

    predicted_results = clf.predict(training_embedding_matrix)

    #Output metrics of predictions
    __output_metrics(clf, training_embedding_matrix, label_matrix, predicted_results)

#Calculate relevant metrics and graphs for classifier performance
def __output_metrics(clf, training_embedding_matrix, label_matrix, predicted_results):
    #Calculate relevant metrics and graph for the classifier
    print(classification_report(label_matrix, predicted_results))
    print(confusion_matrix(label_matrix, predicted_results))
    print("Reciever Operating Characteristic Area Under the Curve Score", roc_auc_score(label_matrix, clf.predict_proba(training_embedding_matrix)[:, 1]))
    plot = RocCurveDisplay.from_predictions(label_matrix, clf.predict_proba(training_embedding_matrix)[:, 1], plot_chance_level = True)
    _ = plot.ax_.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
    title="Trainning AI vs Real Image Detection\nReceiver Operating Characteristic",
    )
    plt.show()  


#Function to begin the testing of the classifer and to produce relevant metrics pertineant to the project
def test_classifier():
    #load in relevant datasets and embedder
    embedder = Embedder() 
    #Obtained from https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney
    #Obtained from https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney
    midjourney_matrix = embed_huggingface_datasets(embedder, "ideepankarsharma2003/AIGeneratedImages_Midjourney", "test", "test_embeddings_midjourney", 500, 128)

    #Images here were obtained from the COCO datast, val2017, using the first 1000 images
    coco_matrix = embed_locally_saved_datasets(embedder, r"C:\Users\zacha\projects\CIS430\Videre\core\val2017", "test_coco_embeddings", 500, 128, 1000, 1500)

    #natural images from https://www.kaggle.com/datasets/prasunroy/natural-images for real images, and 
    #https://www.kaggle.com/datasets/gpch2159/ai-vs-human-syn-imgs-v2-partial/data
    #for ai generated images using stable Diffusion XL
    natural_matrix = embed_locally_saved_datasets(embedder, r"C:\Users\zacha\projects\CIS430\Videre\core\natural_images\embedding_images", "test_natural_embeddings", 500, 128, 0, 500)
    ai_matrix = embed_locally_saved_datasets(embedder, r"C:\Users\zacha\projects\CIS430\Videre\core\stabilityai.stable-diffusion-xl-refiner-1.0_0.5_12_2025.02.25_05.15.08_846327", "test_ai_embeddings", 500, 128, 1000, 1500)

    #create corresponding label arrays for each matrix 
    #then concatenate them into one label matrix that will correspond to the collapsed 
    midjourney_labels = [1] * 500
    coco_labels =  [0] * 497
    natural_image_labels = [0] * 497
    stable_diff_labels = [1] * 497

    #Form one singular matrix of the labels
    label_matrix = np.concatenate((midjourney_labels, coco_labels, natural_image_labels, stable_diff_labels), axis = 0)

    #Create the corresponding embeddings matrix to test the model 
    training_embedding_matrix = np.concatenate((midjourney_matrix, coco_matrix, natural_matrix, ai_matrix), axis = 0)

    #Load in the classifier 
    clf = joblib.load("final_classifier.joblib")

    #Predict results of test data
    predicted_results = clf.predict(training_embedding_matrix)

    #Output metrics of predictions
    __output_metrics(clf, training_embedding_matrix, label_matrix, predicted_results)

def embed_huggingface_datasets(embedder, dataset, training_split, save_file_name, num_images, batch_size):
    images_list = load_dataset(dataset, split = training_split, streaming = True)
    embedding_list = []
    batch = []

    #Batch images into specificed length groups up until the last index
    for index, row in enumerate(images_list):
        if (len(batch) < batch_size and index != num_images):
            batch.append(row["image"])
        if (index == num_images):
            break
        else:
            embedding_list.append(embedder._embed(batch))
            batch.clear()
    #Save the embeddings for temporary use
    embedding_matrix = np.concatenate(embedding_list, axis = 0)
    np.save(save_file_name, embedding_matrix)
    return embedding_matrix 

def embed_locally_saved_datasets(embedder, dataset_path, save_file_name, num_images, batch_size, index_start, index_end):
    #Pull images from local path (Requires adjustment for given users machine) and store them locally 
    images = Path(dataset_path).glob('*.jpg')
    images_list = []
    batch = []
    embedding_list = []
    #grab first 500 images after training data set of 1000
    for index, image in enumerate(islice(images, index_start, index_end)):
        if(index < num_images):
            images_list.append(image)
        else:
            break
    for index, row in enumerate(images_list):
        #batch images into groups of 128 for embedding
        if (len(batch) < batch_size):
            batch.append(Image.open(images_list[index]).convert("RGB"))
        else:
            #embed images
            embedding_list.append(embedder._embed(batch))
            batch.clear()
    if (len(batch) > 0):
        embedding_list.append(embedder._embed(batch))
        batch.clear()
    #Save the embeddings for temporary use
    embedding_matrix = np.concatenate(embedding_list, axis = 0)
    np.save(save_file_name, embedding_matrix)
    return embedding_matrix

#Function to convert user's inputted image
def convert_input_image(input):
    #Instiate embedding object to load relevant models
    embedder = Embedder()
    #Convert incoming image bytes to image and Open image
    bytes_to_image = BytesIO(input)
    image = [Image.open(bytes_to_image)]
    embedding = embedder._embed(image)
    return embedding

#Function for Full stack application to predict image 
def predict_image(input):
    #Convert image to embeddings
    embeddings = convert_input_image(input)

    #Load classifer and predict the image's class and the probabilities of it
    clf = joblib.load("final_classifier.joblib")
    predicted_class = clf.predict(embeddings)
    prob_real, prob_fake = clf.predict_proba(embeddings)[0]
    
    #Create a JSON payload for the API to return, and return the predicted values, typecasting them to regular floats instead of numpy floats
    results_payload = {"class": int(predicted_class[0]), "probability_real": float(prob_real), "probability_ai": float(prob_fake)}
    return results_payload
