from fastapi import routing
from fastapi import FastAPI, File, UploadFile
import numpy as np
from .Application import predict_image

from ultralytics import YOLO
yolo=YOLO("yolov8n.pt")

router=routing.APIRouter()



@router.post("/DogsBreedClassification")
async def predict__(file: UploadFile = File(...)):
    print("Working")
    try:
        image_data = await file.read()
        # Open the image using PIL from the byte data
        
        send_Images,send_Predictions=predict_image(image_data)
        
        print("Predictions Completed")
 
        # Respond with a success message
        return {"message": f"File {file.filename} uploaded and opened successfully!","image":send_Images,"predictions":send_Predictions}
    
    except Exception as e: 
        return {"message": f"Failed to process image. Error: {str(e)}"}
