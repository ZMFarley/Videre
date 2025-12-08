# IMPORT SECTION
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from embedding import convert_input_image
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
@app.post("/upload-image")
async def upload_image():
    print(convert_input_image)