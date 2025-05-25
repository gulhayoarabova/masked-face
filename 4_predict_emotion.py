import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class_names = ['happy', 'angry', 'neutral', 'sad', 'surprised']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class EmotionNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionNet, self).__init__()
        self.base = models.resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionNet(num_classes=5).to(device)
model.load_state_dict(torch.load('emotion_model.pth'))
model.eval()

img_path = input("Enter the image path: ")
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    _, pred = torch.max(output, 1)
    predicted_emotion = class_names[pred.item()]
    print(f"🤔 The predicted emotion is: {predicted_emotion}")
