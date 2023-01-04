import torch
import torchvision
from torchvision import transforms, datasets

model = torchvision.models.resnet50(pretrained=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.ImageNet(root="ImageNet_CNN/data", train=False, transforms=transform)