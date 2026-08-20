# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET

Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model

![alt text](img/nnm.png)

## DESIGN STEPS
### STEP 1: 

Load the Fashion-MNIST training and testing datasets and preprocess the images by converting them into tensors and normalizing the pixel values.

### STEP 2: 

Create DataLoaders to divide the datasets into batches for efficient training and testing.



### STEP 3: 

Build a CNN model with three convolutional layers, ReLU activation, max-pooling layers, and fully connected layers for classifying images into 10 categories.

### STEP 4: 

Initialize the Cross-Entropy Loss function and Adam optimizer, then train the CNN for multiple epochs using forward propagation, loss calculation, backpropagation, and weight updates.

### STEP 5: 

Evaluate the trained model using the test dataset and calculate the classification accuracy.


### STEP 6: 

Generate a confusion matrix and classification report to analyze the model's prediction performance using precision, recall, and F1-score.

## PROGRAM

### Name: YUVARAM S

### Register Number: 212224230315

```python


# ================================
# 1. Import Required Libraries
# ================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ================================
# 2. Data Preprocessing and Loading
# ================================

# Convert images to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load training dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

# Load test dataset
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# Check dataset details
image, label = train_dataset[0]
print("Train Image Shape:", image.shape)
print("Train Dataset Size:", len(train_dataset))

image, label = test_dataset[0]
print("Test Image Shape:", image.shape)
print("Test Dataset Size:", len(test_dataset))

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ================================
# 3. Define CNN Model
# ================================

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # Convolution layers
        self.c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.pool(torch.relu(self.c3(x)))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ================================
# 4. Initialize Model, Loss, Optimizer
# ================================

# Create model
model = CNNClassifier()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print model summary
summary(model, input_size=(1, 28, 28))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ================================
# 5. Train the Model
# ================================

def train_model(model, train_loader, num_epochs=3):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


# Train the model
train_model(model, train_loader)


# ================================
# 6. Test the Model
# ================================

def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradients
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=test_dataset.classes))


# Evaluate model
test_model(model, test_loader)


# ================================
# 7. Predict on a Single Image
# ================================

def predict_image(model, image_index, dataset):
    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():
        image = image.to(device)
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)

    class_names = dataset.classes

    # Display image
    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f"Actual: {class_names[label]} | Predicted: {class_names[predicted.item()]}")
    plt.axis("off")
    plt.show()

    print(f"Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}")


# Example prediction
predict_image(model, image_index=80, dataset=test_dataset)



```

### OUTPUT

## Training Loss per Epoch

![alt text](img/epoch.png)

## Accuracy

![alt text](img/accuracy.png)

## Confusion Matrix

![alt text](<img/confusion matrix.png>)

## Classification Report

![alt text](<img/classification report.png>)

## summary

![alt text](img/summary.png)

### New Sample Data Prediction

![alt text](<img/new predicted sample data.png>)

## RESULT

Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
