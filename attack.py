import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from simpleFCN import SimpleFCN
import matplotlib.pyplot as plt

model = torch.load("model.pth")

criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform)

total = 0
success = 0

for image, label in dataset:
    total += 1
    print("Label:", label, end=" ")

    # Display the original image
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.show()

    image.requires_grad_()

    target_tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]])

    epsilon = 0.0004

    alpha = 0.01

    for i in range(1000):
        output = model(image)
        loss = criterion(output, target_tensor)
        loss.backward()

        perturbation = alpha * image.grad.sign()
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        image.data = image.data - perturbation

    output = model(image)
    prediction = output.argmax().item()
    if prediction == 8:
        success += 1
    print("Prediction:", prediction)

    # Display the modified image
    plt.imshow(image.squeeze().detach().numpy(), cmap='gray')
    plt.show()
    break

print("Advantage: ", success / total)
