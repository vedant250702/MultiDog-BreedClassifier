from ultralytics import YOLO
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from PIL import Image
import base64
import io



# Loading the model
checkpoint=torch.load("./model/best_model50.pth")
model=models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features,120)
model.load_state_dict(checkpoint["model_state_dict"])


device="cpu"
yolo=YOLO("yolov8n.pt")
model.to(device)

# Loading the classes of the dogs.
class_names = None
with open("./classes.json","r") as file:
    class_names=json.load(file)
class_names=class_names["classes"]


# This function predicts the breed of the dog.
def classify(image, model, class_names):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    # print(predicted.item())
    predicted_class = class_names[int(predicted.item())]
    return predicted_class




# This function involves detecting multiple dogs and cropping them.
# After cropping it is sent to classify(image, model, class_names) function.
def predict_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_array=np.array(image)
        # print(image.shape)
    results = yolo(image,device="cpu")

    dog_class_id = 16

    send_Images=[]
    send_Predictions=[]
    for i, box in enumerate(results[0].boxes.data):
        x1, y1, x2, y2, conf, class_id = box
        if int(class_id) == dog_class_id:
            cropped_dog = image_array[int(y1):int(y2), int(x1):int(x2)]
            cropped_dog=Image.fromarray(cropped_dog)
            prediction=classify(cropped_dog,model,class_names)

            buffered = io.BytesIO()
            cropped_dog.save(buffered, format="JPEG")  # Save PIL image to buffer
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # Convert to base64 string
            send_Predictions.append(prediction)
            send_Images.append(img_str)
    return send_Images,send_Predictions