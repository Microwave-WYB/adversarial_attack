import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

def onehot(num):
    output = [0.0] * 10
    output[num] += 1
    return output

device = torch.device("cuda")
model = torch.load("MINIST_CNN/model.pth")

criterion = nn.CrossEntropyLoss()

dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

total = 0
success = 0

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

target = 8
target_tensor = torch.tensor([onehot(target)]).to(device)

for i, (image, label) in enumerate(data_loader):

    # Display the original image
    # plt.imshow(image.squeeze().numpy(), cmap='gray')
    # plt.show()

    image = image.to(device)
    total += 1
    print("Label:", label.item(), end=" ")

    image.requires_grad_()

    for i in range(100):
        optimizer = optim.SGD([image], lr = 2e-4, momentum=0.9)
        # optimizer = optim.Adam([image], 0.003)
        output = model(image)
        loss = criterion(output, target_tensor).to(device)
        loss.backward()
        optimizer.step()

    output = model(image)
    prediction = output.argmax().item()
    if prediction == target:
        success += 1
    print("Prediction:", prediction)

    # Display the modified image
    # plt.imshow(image.cpu().squeeze().detach().numpy(), cmap='gray')
    # plt.show()

    print("Advantage: ", success / total)
