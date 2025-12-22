# IMPORT SECTION
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from embedding import predict_image
from io import BytesIO
from PIL import Image
app = FastAPI()

#Pydantic Model to validate incoming result data
class Prediction(BaseModel):
    result: int
    probability_real: float
    probability_ai: float

# ACCEPTABLE ORIGINS
origins = ["http://localhost:5173", "http://127.0.0.1:5173"] 
# CORS_MIDDLEWARE TO PREVENT CORS ISSUE
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# UPDATES NORMAL TRAFFIC SECTION OF THE DASHBOARD 
@app.post("/predict", response_model=Prediction)
async def predict_image_class(file: UploadFile = File(...)) -> Prediction:
    #Validate image is uncorrupted and is a valid image
    try:
        Image.open(file.file).verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Image corrupted or invalid")
    file.file.seek(0)

    #Read in value as bytes for passing to predictor
    input = await file.read()

    #Validate image has arrived before continuing
    if not input:
          raise HTTPException(status_code=400, detail="No image recieved")
    #Attempt prediction, throw error during failure
    try:
        prediction = predict_image(input)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to Predict Image")
    
    #Validate Result fits 1 or 0 (real or fake)
    if prediction["result"] != 0 or prediction["result"] != 1:
        HTTPException(status_code=500, detail="Invalid prediction result") 
    #Validate Probabilities fall within proper range, 0-100%
    if not 0 <= prediction["probability_real"] <= 1:
        raise HTTPException(status_code=500, detail="Invalid prediction result") 
    
    if not 0 <= prediction["probability_ai"] <= 1:
        raise HTTPException(status_code=500, detail="Invalid prediction result") 
    
    return prediction
    