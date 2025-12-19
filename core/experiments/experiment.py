from core.embedding import Embedder, embed_huggingface_datasets, embed_locally_saved_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import joblib
#This file is directly intended for the creation and testing of different classifiers and embeddings, allowing for
#A more modular approach to the creation and testing of individual portions


#Calculate relevant metrics and graphs for classifier performance
def __output_metrics(clf: LogisticRegression, training_embedding_matrix: np.ndarray, label_matrix: np.ndarray, predicted_results: np.ndarray) -> None:
    #Calculate relevant metrics and graph for the classifier
    print(classification_report(label_matrix, predicted_results))
    print(confusion_matrix(label_matrix, predicted_results))
    print("Reciever Operating Characteristic Area Under the Curve Score", roc_auc_score(label_matrix, clf.predict_proba(training_embedding_matrix)[:, 1]))
    plot = RocCurveDisplay.from_predictions(label_matrix, clf.predict_proba(training_embedding_matrix)[:, 1], plot_chance_level = True)
    _ = plot.ax_.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
    title="Trainning AI vs Real Image Detection\nReceiver Operating Characteristic",
    )
    plt.show()

def create_embeddings_and_classifier() -> None:
    #load in relevant datasets and embedder
    embedder = Embedder()
    #Obtained from https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney
    midjourney_matrix = embed_huggingface_datasets(embedder, "ideepankarsharma2003/AIGeneratedImages_Midjourney", "train", "embeddings_midjourney", 1000, 192)
    if midjourney_matrix.shape[0] == 0:
        raise ValueError("Midjourney HF Matrix empty")
    #Images here were obtained from the COCO datast, val2017, using the first 1000 images
    coco_matrix = embed_locally_saved_datasets(embedder, r"core\experiments\datasets\val2017", "embeddings_coco", 1000, 192, 0, 1000)
    if coco_matrix.shape[0] == 0:
        raise ValueError("COCO val2017 Matrix empty")
    #natural images from https://www.kaggle.com/datasets/prasunroy/natural-images for real images, and 
    #https://www.kaggle.com/datasets/gpch2159/ai-vs-human-syn-imgs-v2-partial/data
    #for ai generated images using stable Diffusion XL
    natural_matrix = embed_locally_saved_datasets(embedder, r"core\experiments\datasets\natural_images\embedding_images", "embeddings_natural_images", 1000, 192, 0, 1000)
    if natural_matrix.shape[0] == 0:
        raise ValueError("Natural Images Matrix empty")
    
    ai_matrix = embed_locally_saved_datasets(embedder, r"core\experiments\datasets\stabilityai.stable-diffusion-xl-refiner-1.0_0.5_12_2025.02.25_05.15.08_846327", "embeddings_stableDiffXL", 1000, 192, 0, 1000)
    if ai_matrix.shape[0] == 0:
        raise ValueError("StableDifussionXL Matrix empty")
    
    #create corresponding label arrays for each matrix 
    #then concatenate them into one label matrix that will correspond to the collapsed 
    midjourney_labels = [1] * 1000
    coco_labels =  [0] * 1000
    natural_image_labels = [0] * 1000
    stable_diff_labels = [1] * 1000

    label_matrix = np.concatenate((midjourney_labels, coco_labels, natural_image_labels, stable_diff_labels), axis = 0)

    #Create the corresponding embeddings matrix to train the model 
    training_embedding_matrix = np.concatenate((midjourney_matrix, coco_matrix, natural_matrix, ai_matrix), axis = 0)

    #Ensure proper matrix shape and contents before classifier training
    if label_matrix.shape[0] != training_embedding_matrix.shape[0]:
        raise ValueError(f"Training Matrix shape: {[training_embedding_matrix.shape[0]]} != Label Matrix shape: {[label_matrix.shape[0]]}")

    if not np.isfinite(training_embedding_matrix).all():
        raise ValueError("Training Matrix contains NaN or inf values")
      
    #Create the logistic regression classifier, and train it on the relevant data, saving it afterwards 
    clf = LogisticRegression()

    clf.fit(training_embedding_matrix, label_matrix)

    #joblib.dump(clf, "test_classifier.joblib")

    predicted_results = clf.predict(training_embedding_matrix)

    #Output metrics of predictions
    __output_metrics(clf, training_embedding_matrix, label_matrix, predicted_results)

#Function to begin the testing of the classifer and to produce relevant metrics pertineant to the project
def test_classifier() -> None:
    #load in relevant datasets and embedder
    embedder = Embedder() 
    #Obtained from https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney
    midjourney_matrix = embed_huggingface_datasets(embedder, "ideepankarsharma2003/AIGeneratedImages_Midjourney", "test", "test_embeddings_midjourney", 500, 192)
    if midjourney_matrix.shape[0] == 0:
        raise ValueError("Midjourney HF Matrix empty")
    #Images here were obtained from the COCO datast, val2017, using the first 1000 images
    coco_matrix = embed_locally_saved_datasets(embedder, r"core\experiments\datasets\val2017", "test_coco_embeddings", 500, 192, 1000, 1500)
    if coco_matrix.shape[0] == 0:
        raise ValueError("Coco Val2017 Matrix empty")
    
    #natural images from https://www.kaggle.com/datasets/prasunroy/natural-images for real images, and 
    #https://www.kaggle.com/datasets/gpch2159/ai-vs-human-syn-imgs-v2-partial/data
    #for ai generated images using stable Diffusion XL
    natural_matrix = embed_locally_saved_datasets(embedder, r"core\experiments\datasets\natural_images\test_images", "test_natural_embeddings", 500, 192, 0, 500)
    if natural_matrix.shape[0] == 0:
        raise ValueError("Natural Images Matrix empty")
        
    ai_matrix = embed_locally_saved_datasets(embedder, r"core\experiments\datasets\stabilityai.stable-diffusion-xl-refiner-1.0_0.5_12_2025.02.25_05.15.08_846327", "test_ai_embeddings", 500, 192, 1000, 1500)
    if natural_matrix.shape[0] == 0:
        raise ValueError("StableDifussionXL Matrix empty")

    #create corresponding label arrays for each matrix 
    #then concatenate them into one label matrix that will correspond to the collapsed 
    midjourney_labels = [1] * 500
    coco_labels =  [0] * 500
    natural_image_labels = [0] * 500
    stable_diff_labels = [1] * 500

    #Form one singular matrix of the labels
    label_matrix = np.concatenate((midjourney_labels, coco_labels, natural_image_labels, stable_diff_labels), axis = 0)

    #Create the corresponding embeddings matrix to test the model 
    training_embedding_matrix = np.concatenate((midjourney_matrix, coco_matrix, natural_matrix, ai_matrix), axis = 0)

    #Check to confirm matrices are of same shape and proper values before training classifier
    if label_matrix.shape[0] != training_embedding_matrix.shape[0]:
        raise ValueError(f"Training Matrix shape: {[training_embedding_matrix.shape[0]]} != Label Matrix shape: {[label_matrix.shape[0]]}")
    
    if not np.isfinite(training_embedding_matrix).all():
        raise ValueError("Training Matrix contains NaN or inf values")
    
    #Load in the classifier 
    clf = joblib.load(r"core\final_classifier.joblib")

    #Predict results of test data
    predicted_results = clf.predict(training_embedding_matrix)

    #Output metrics of predictions
    __output_metrics(clf, training_embedding_matrix, label_matrix, predicted_results)

