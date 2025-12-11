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

#Loading Pretrained/Frozen embedders
model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast = True)

#Load in Relevant GPU or CPU to run model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def create_embeddings_and_classifier():
    #load in relevant datasets
    #This is the saved code used to load in my corresponding datasets. They were frozen in comments for later refactoring while i 
    #Began iterating the progress on my overarching project 
    #Obtained from https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney
    midjourney_ai_images = load_dataset("ideepankarsharma2003/AIGeneratedImages_Midjourney", split ="train", streaming = True)
    midjourney_list = []
    batch = []

    #Batch images into 128 groups up until the last index
    for index, row in enumerate(midjourney_ai_images):
        if (len(batch) < 128 and index != 1000):
            batch.append(row["image"])
        if (index == 1000):
            break
        else:
            #Convert images into embeddings
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            midjourney_list.append(embeddings)
            batch.clear()
    #Save the embeddings for temporary use
    midjourney_matrix = np.concatenate(midjourney_list, axis = 0)
    np.save("embeddings_midjourney", midjourney_matrix)

    #Pull images from local path (Requires adjustment for given users machine) and store them locally 
    coco_images = Path(r"C:\Users\zacha\projects\CIS430\Videre\core\val2017").glob('*.jpg')
    coco_list = []
    #grab first 1000 images
    for index, image in enumerate(coco_images):
        if(index < 1000):
            coco_list.append(image)
        else:
            break
    batch = []
    coco_embedding_list = []
    #Batch images into 128 groups up until the last index
    for index, row in enumerate(coco_list):
        if (len(batch) < 128):
            batch.append(Image.open(coco_list[index]).convert("RGB"))
        else:
            #Convert to vector embeddings
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            coco_embedding_list.append(embeddings)
            batch.clear()
    #Embed remaining images
    if (len(batch) > 0):
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Convert the tensor to a numpy array for storage 
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        coco_embedding_list.append(embeddings)
        batch.clear()
    #Save the embeddings for temporary use
    coco_matrix = np.concatenate(coco_embedding_list, axis = 0)
    np.save("embeddings_coco", coco_matrix)

    #ended up not using CIFAKE from Kaggle due to how small the images were, concerns on comparison to other datasets
    #instead combined natural images from https://www.kaggle.com/datasets/prasunroy/natural-images for real images, and 
    #https://www.kaggle.com/datasets/gpch2159/ai-vs-human-syn-imgs-v2-partial/data
    #for ai generated images using stable Diffusion XL

    #Pull images from local path (Requires adjustment for given users machine) and store them locally 
    natural_images = Path(r"C:\Users\zacha\projects\CIS430\Videre\core\natural_images\embedding_images").glob('*.jpg')
    natural_image_list = []
    for index, image in enumerate(natural_images):
        if(index < 1000):
            natural_image_list.append(image)
        else:
            break
    batch = []
    embeddings_natural_list = []
    for index, row in enumerate(natural_image_list):
        #batch images into groups of 128
        if (len(batch) < 128):
            batch.append(Image.open(natural_image_list[index]).convert("RGB"))
        else:
            #embed images
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            embeddings_natural_list.append(embeddings)
            batch.clear()
    #embed remaining images
    if (len(batch) > 0):
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Convert the tensor to a numpy array for storage 
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        embeddings_natural_list.append(embeddings)
        batch.clear()
    #Save the embeddings for temporary use
    natural_matrix = np.concatenate(embeddings_natural_list, axis = 0)
    np.save("embeddings_natural_images", natural_matrix)

    #Pull images from local path (Requires adjustment for given users machine) and store them locally 
    ai_images = Path(r"C:\Users\zacha\projects\CIS430\Videre\core\stabilityai.stable-diffusion-xl-refiner-1.0_0.5_12_2025.02.25_05.15.08_846327").glob('*.jpg')
    ai_image_list = []
    for index, image in enumerate(ai_images):
        #grab 1000 images for embedding
        if(index < 1000):
            ai_image_list.append(image)
        else:
            break
    batch = []
    embeddings_ai_list = []
    for index, row in enumerate(ai_image_list):
        #batch images into groups of 128 for embedding
        if (len(batch) < 128):
            batch.append(Image.open(ai_image_list[index]).convert("RGB"))
        else:
            #embed the images
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            embeddings_ai_list.append(embeddings)
            batch.clear()
        #embed remaining images
    if (len(batch) > 0):
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Convert the tensor to a numpy array for storage 
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        embeddings_ai_list.append(embeddings)
        batch.clear()
    #Save the embeddings for temporary use
    ai_matrix = np.concatenate(embeddings_ai_list, axis = 0)
    np.save("embeddings_stableDiffXL", ai_matrix)

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

    joblib.dump(clf, "test_classifier.joblib")

#Function to begin the testing of the classifer and to produce relevant metrics pertineant to the project
def test_classifier():
    #load in relevant datasets
    #This is the saved code used to load in my corresponding datasets. They were frozen in comments for later refactoring while i 
    #Began iterating the progress on my overarching project 
    #Obtained from https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney
    midjourney_ai_images = load_dataset("ideepankarsharma2003/AIGeneratedImages_Midjourney", split ="test", streaming = True)
    midjourney_list = []
    batch = []
    #grab first 500 images for testing
    for index, row in enumerate(midjourney_ai_images):
        if (len(batch) < 128 and index != 500):
            batch.append(row["image"])
        if (index == 500):
            break
        else:
            #embed images
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            midjourney_list.append(embeddings)
            batch.clear()
    #Save the embeddings for temporary use
    midjourney_matrix = np.concatenate(midjourney_list, axis = 0)
    np.save("test_midjourney_embeddings", midjourney_matrix)

    #Pull images from local path (Requires adjustment for given users machine) and store them locally 
    coco_images = Path(r"C:\Users\zacha\projects\CIS430\Videre\core\val2017").glob('*.jpg')
    coco_list = []
    #grab first 500 images after training data set of 1000
    for index, image in enumerate(islice(coco_images, 1000, 1500)):
        if(index < 500):
            coco_list.append(image)
        else:
            break
    batch = []
    coco_embedding_list = []
    for index, row in enumerate(coco_list):
        #batch images into groups of 128 for embedding
        if (len(batch) < 128):
            batch.append(Image.open(coco_list[index]).convert("RGB"))
        else:
            #embed images
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            coco_embedding_list.append(embeddings)
            batch.clear()
    #embed remaining images
    if (len(batch) > 0):
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Convert the tensor to a numpy array for storage 
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        coco_embedding_list.append(embeddings)
        batch.clear()
    #Save the embeddings for temporary use
    coco_matrix = np.concatenate(coco_embedding_list, axis = 0)
    np.save("test_coco_embeddings", coco_matrix)

    #Pull images from local path (Requires adjustment for given users machine) and store them locally 
    natural_images = Path(r"C:\Users\zacha\projects\CIS430\Videre\core\natural_images\test_images").glob('*.jpg')
    natural_image_list = []
    #grab 500 images for testing
    for index, image in enumerate(natural_images):
        if(index < 500):
            natural_image_list.append(image)
        else:
            break
    batch = []
    embeddings_natural_list = []
    for index, row in enumerate(natural_image_list):
        #batch images into groups of 128
        if (len(batch) < 128):
            batch.append(Image.open(natural_image_list[index]).convert("RGB"))
        else:
            #embed images
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            embeddings_natural_list.append(embeddings)
            batch.clear()
    #Embed remaining images
    if (len(batch) > 0):
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Convert the tensor to a numpy array for storage 
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        embeddings_natural_list.append(embeddings)
        batch.clear()
    #Save the embeddings for temporary use
    natural_matrix = np.concatenate(embeddings_natural_list, axis = 0)
    np.save("test_natural_embeddings", natural_matrix)

    #Pull images from local path (Requires adjustment for given users machine) and store them locally 
    ai_images = Path(r"C:\Users\zacha\projects\CIS430\Videre\core\stabilityai.stable-diffusion-xl-refiner-1.0_0.5_12_2025.02.25_05.15.08_846327").glob('*.jpg')
    ai_image_list = []
    #Grab first 500 images after the original training data set
    for index, image in enumerate(islice(ai_images, 1000, 1500)):
        if(index < 500):
            ai_image_list.append(image)
        else:
            break
    batch = []
    embeddings_ai_list = []
    #batch images into groups of 128 and embed them 
    for index, row in enumerate(ai_image_list):
        if (len(batch) < 128):
            batch.append(Image.open(ai_image_list[index]).convert("RGB"))
        else:
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
            #Convert the tensor to a numpy array for storage 
            embeddings = outputs.pooler_output.detach().cpu().numpy()
            embeddings_ai_list.append(embeddings)
            batch.clear()
    #Embed remaining images
    if (len(batch) > 0):
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Convert the tensor to a numpy array for storage 
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        embeddings_ai_list.append(embeddings)
        batch.clear()
    #Save the embeddings for temporary use
    ai_matrix = np.concatenate(embeddings_ai_list, axis = 0)
    np.save("test_ai_embeddings", ai_matrix)

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

    #Calculate relevant metrics and graph for the classifier
    print(classification_report(label_matrix, predicted_results))
    print(confusion_matrix(label_matrix, predicted_results))
    print("Reciever Operating Characteristic Area Under the Curve Score", roc_auc_score(label_matrix, clf.predict_proba(training_embedding_matrix)[:, 1]))
    plot = RocCurveDisplay.from_predictions(label_matrix, clf.predict_proba(training_embedding_matrix)[:, 1], plot_chance_level = True)
    _ = plot.ax_.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
    title="AI vs Real Image Detection\nReceiver Operating Characteristic",
)
    plt.show()
#Helper function to load in model 
def __load_model():
    #Loading Pretrained/Frozen embedders
    model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast = True)

    #Load in Relevant GPU or CPU to run model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, processor

#Function to convert user's inputted image
def convert_input_image(input):
    #Load in embedder
    model, processor = __load_model()

    #Convert incoming image bytes to image and Open image
    bytes_to_image = BytesIO(input)
    image = Image.open(bytes_to_image)
    #Disables gradient calculation as we are only embedding vectors, no need for model training
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

    #Convert the tensor to a numpy array for storage 
    embeddings = outputs.pooler_output.detach().cpu().numpy()
    return embeddings

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
