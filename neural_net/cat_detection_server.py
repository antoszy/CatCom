from model import CatDetector
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

        async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            # Async write file to temp directory
            async for chunk in file.file:
                await tmp.write(chunk)
            tmp_name = tmp.name

        is_cat_meowing = await self.cat_detector.detect_cat(tmp_name)
        
        os.remove(tmp_name)

        return {"cat_meowing_detected": is_cat_meowing}

# Initialize FastAPI app
app = FastAPI()

# Create an instance of the API handler
api_handler = CatDetectorAPI()

# Define the route using the handler method
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await api_handler.upload_file(file)
