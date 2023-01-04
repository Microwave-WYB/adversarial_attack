import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import simpleCNN
from tqdm import tqdm

device = torch.device("cuda")

# define the model
model = simpleCNN.SimpleCNN().to(device)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# load the MNIST training and validating datasets
train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
val_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# create dataloaders to feed the data to the training and test loops
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# define the number of epochs to train the model
num_epochs = 10

best_loss = 99999999999;
# train the model
for epoch in tqdm(range(num_epochs)):
    # initialize the running loss for the epoch
    running_loss = 0.0
    
    # set the model to train mode
    model.train()
    
    # loop over the training data
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # update the running loss
        running_loss += loss.item()
        
    # print the loss for the epoch
    print(f'Epoch {epoch+1}: training loss = {running_loss / len(train_loader)}')
    
    # set the model to evaluation mode
    model.eval()
    
    # initialize the running loss and accuracy for the epoch
    running_loss = 0.0
    running_acc = 0.0
    
    # loop over the val data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # update the running loss and accuracy
            running_loss += loss.item()
            running_acc += (outputs.argmax(dim=1) == labels).float().mean()
            
    # print the loss and accuracy for the epoch
    print(f'Epoch {epoch+1}: validation loss = {running_loss / len(val_loader)}, validation accuracy = {running_acc / len(val_loader)}')

    # update best loss and save model
    if (running_loss < best_loss):
        best_loss = running_loss
        torch.save(model, "MINIST_CNN/model.pth")
    

