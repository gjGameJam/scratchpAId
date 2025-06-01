import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm  # for progress bars during training

# define a simple convolutional neural network for image classification
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # first convolutional layer: 1 input channel (grayscale), 16 output channels, 3x3 kernel, padding for same size
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # downsample by 2x

        # second convolutional layer: 16 input channels, 32 output channels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # downsample again

        # fully connected layers for classification
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # input from flattened feature maps
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        # forward pass: conv -> relu -> pool -> conv -> relu -> pool -> flatten -> fc -> relu -> fc
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten tensor for fc layer
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# load mnist dataset and return train/test loaders
# this function encapsulates dataset loading logic and applies a simple ToTensor transform
# it returns dataloaders for convenient iteration over the data in batches

def get_data_loaders(batch_size=64):
    # convert PIL images to tensor format scaled between 0 and 1
    transform = transforms.ToTensor()

    # download and prepare training dataset with transformation
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # download and prepare test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # wrap datasets in DataLoader for efficient batch access
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# train the model for the specified number of epochs
# loops through the training set, computes loss, performs backpropagation, updates weights
# prints training loss and accuracy along with test accuracy for each epoch

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        running_loss, correct, total = 0.0, 0, 0

        # wrap data loader with tqdm for progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            # forward pass: compute predictions and loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass: compute gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss and correct predictions for metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # summarize training results for the epoch
        epoch_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        # evaluate the model on test data
        test_accuracy = evaluate_model(model, test_loader, device)

        # print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")

# evaluate model performance on test set
# switches to eval mode, disables gradient computation, returns classification accuracy

def evaluate_model(model, test_loader, device):
    model.eval()  # set model to evaluation mode
    correct, total = 0, 0

    # no gradients needed for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total

# visualize a grid of predictions on the test set
# displays up to 100 test images with predicted and actual labels
# marks correct predictions with green ✓ and incorrect ones with red ✗
# this function helps human-audit model predictions and identify failure cases

def audit_predictions(model, test_loader, device, max_images=100):
    model.eval()  # set model to evaluation mode
    correct, total, shown = 0, 0, 0

    # prepare subplot grid: 10x10 for max 100 images
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.flatten()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(images.size(0)):
                if shown >= max_images:
                    break

                # extract image and prediction info
                img = images[i].cpu().squeeze(0).numpy()
                pred, true = preds[i].item(), labels[i].item()

                # green check for correct, red x for wrong
                color = 'green' if pred == true else 'red'
                mark = '✓' if pred == true else '✗'

                # set image title with color-coded prediction status
                title = f"P:{pred} / A:{true} {mark}"
                ax = axes[shown]
                ax.imshow(img, cmap='gray')
                ax.set_title(title, fontsize=8, color=color)
                ax.axis("off")

                # update stats
                correct += int(pred == true)
                total += 1
                shown += 1

            if shown >= max_images:
                break

    # finalize and display plot
    plt.tight_layout()
    plt.suptitle(f"Visual Audit | Accuracy: {100.0 * correct / total:.2f}%", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

# entry point of the script
# sets up training and evaluation using MNIST data

if __name__ == "__main__":
    # select device: use cuda if available, otherwise fallback to cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize dataloaders for training and test sets
    train_loader, test_loader = get_data_loaders()

    # initialize model, loss function, and optimizer
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train the model over multiple epochs
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=5)

    # audit predictions with a 10x10 visual grid
    audit_predictions(model, test_loader, device)
