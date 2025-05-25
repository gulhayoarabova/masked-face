import os
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

torch.manual_seed(123)

dataset_path = "dataset/train/"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

torch.save(train_dataset, 'train_data.pt')
torch.save(test_dataset, 'test_data.pt')

print("✅ Data prepared and saved!")
