# IMPORT SECTION
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from embedding import predict_image
from io import BytesIO
from PIL import Image
app = FastAPI()

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
@app.post("/predict")
async def predict_image_class(file: UploadFile = File(...)):
    input = await file.read()
    return predict_image(input)
    