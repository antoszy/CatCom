from model import CatDetector, load_wav_16k_mono_into_tensor
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import aiofiles
import os
import wave


class CatDetectorAPI:
    def __init__(self):
        self.cat_detector = CatDetector()

    async def upload_file(self, file: UploadFile = File(...)):
        if not file.filename.endswith('.wav'):
            return JSONResponse(content={"error": "Invalid file format"}, status_code=400)

        # Read the uploaded file into memory
        contents = await file.read()

        # Convert the file contents to a tensor and resample
        wav_tensor = load_wav_16k_mono_into_tensor(contents)

        # Use the tensor to detect cat meows
        is_cat_meowing = await self.cat_detector.detect_cat(wav_tensor)
        
        return {"cat_meowing_detected": is_cat_meowing}


# Initialize FastAPI app
app = FastAPI()

# Create an instance of the API handler
api_handler = CatDetectorAPI()

# Define the route using the handler method
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await api_handler.upload_file(file)
