import os
import torch
from PIL import Image
from torchvision import transforms
from recognition_model import EmotionRecognitionCNN

# Load model
device = torch.device("cpu")
model = EmotionRecognitionCNN()
model.load_state_dict(torch.load("emotion_recognition_cnn.pth", map_location=device))
model.eval()

# Prepare single image
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img_dir = os.path.join("..", "..", "single_images")
img_path = os.path.join(img_dir, "test1.jpg")
img = Image.open(img_path)
img_tensor = transform(img).unsqueeze(0).to(device)

# Classes
classes = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Predict
outputs = model(img_tensor)
_, predicted = torch.max(outputs, 1)
emotion = classes[predicted.item()]
print("Predicted emotion:", emotion)