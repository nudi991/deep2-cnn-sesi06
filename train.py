import os
import kagglehub
from pathlib import Path
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import random
import numpy as np
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings


DATASET_PATH = Path("/kaggle/input/microsoft-catsvsdogs-dataset")
CAT_PATH = DATASET_PATH / "PetImages" / "Cat"
DOG_PATH = DATASET_PATH / "PetImages" / "Dog"

class DatasetManager:
    def __init__(self):
        self.datasetPath = DATASET_PATH
        self.catPath = CAT_PATH
        self.dogPath = DOG_PATH
        self.catFiles = []
        self.dogFiles = []
        self.imageSizes = Counter()
        self.corruptFiles = []
        self.trainFiles = []
        self.validationFiles = []
        self.testFiles = []
        self.transform = None
        self.trainLoader = None
        self.validationLoader = None
        self.testLoader = None        
        
    def downloadDataset(self):
        print("[Download dataset]")
        if not os.path.exists('data'):
            print("Dataset not found. starting download...")
            kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset", output_dir=self.datasetPath)
        else:
            print("Dataset already exists. therefore we skip...")

    def inspectDataset(self):
        print("\n[Dataset Structure]")

        for item in self.datasetPath.iterdir():
            print(f"- {item}")

        self.catFiles = list(self.catPath.glob("*.jpg"))
        self.dogFiles = list(self.dogPath.glob("*.jpg"))
        
        print(f"- Cat images: {len(self.catFiles)}")
        print(f"- Dog images: {len(self.dogFiles)}")
        print(f"- Total images: {len(self.catFiles) + len(self.dogFiles)}")

    def checkImageFiles(self):
        print("\n[Image Size]")

        for imagePath in self.catFiles + self.dogFiles:
            try:
                with warnings.catch_warnings(record=True) as warningList:
                    warnings.simplefilter("always")

                    with Image.open(imagePath) as image:
                        imageSize = image.size
                        image.load()

                    for warning in warningList:
                        if "Truncated File Read" in str(warning.message):
                            self.corruptFiles.append(imagePath)
                            print(f"- Truncated File Read: {imagePath}")
                            break

                self.imageSizes[imageSize] += 1

            except Exception:
                self.corruptFiles.append(imagePath)

        print(f"Unique image sizes: {len(self.imageSizes)}")
        print("Top 10 most common image sizes:")

        for imageSize, count in self.imageSizes.most_common(10):
            print(f"- {imageSize}: {count}")

        print("\n[Corrupt Images]")
        print(f"Corrupt images: {len(self.corruptFiles)}")

        for imagePath in self.corruptFiles:
            print(f"- {imagePath}")
            
    def splitDataset(self):
        print("\n[Dataset Split]")
        validFiles = [imagePath for imagePath in self.catFiles + self.dogFiles if imagePath not in self.corruptFiles]
        labels = [imagePath.parent.name for imagePath in validFiles]

        self.trainFiles, tempFiles, trainLabels, tempLabels = train_test_split(validFiles, labels, test_size=0.30, random_state=42, stratify=labels)
        self.validationFiles, self.testFiles = train_test_split(tempFiles, test_size=0.50, random_state=42, stratify=tempLabels)
        
        print(f"- Valid images: {len(validFiles)}")
        print(f"- Train images: {len(self.trainFiles)}")
        print(f"- Validation images: {len(self.validationFiles)}")
        print(f"- Test images: {len(self.testFiles)}")

    def createTransforms(self):
        print("\n[Preprocessing]")
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        print("- Resize: 128x128")
        print("- Color channels: RGB")
        print("- Pixel normalization: 0-1")

    def createDataLoaders(self):
        print("\n[DataLoader]")
        trainDataset = CatsDogsDataset(self.trainFiles, self.transform)
        validationDataset = CatsDogsDataset(self.validationFiles, self.transform)
        testDataset = CatsDogsDataset(self.testFiles, self.transform)

        self.trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
        self.validationLoader = DataLoader(validationDataset, batch_size=32, shuffle=False)
        self.testLoader = DataLoader(testDataset, batch_size=32, shuffle=False)

        print(f"- Train batches: {len(self.trainLoader)}")
        print(f"- Validation batches: {len(self.validationLoader)}")
        print(f"- Test batches: {len(self.testLoader)}")

class CatsDogsDataset(Dataset):
    def __init__(self, imageFiles, transform=None):
        self.imageFiles = imageFiles
        self.transform = transform

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):

        imagePath = self.imageFiles[index]
        label = 0 if imagePath.parent.name == "Cat" else 1

        image = Image.open(imagePath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def setSeed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, loader, criterion, optimizer, device):
    model.train()
    totalLoss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        totalLoss += loss.item() * images.size(0)

        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    averageLoss = totalLoss / total
    accuracy = correct / total

    return averageLoss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    totalLoss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            totalLoss += loss.item() * images.size(0)

            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    averageLoss = totalLoss / total
    accuracy = correct / total

    return averageLoss, accuracy

def trainModel(model, trainLoader, validationLoader, criterion, optimizer, device, epochs):
    print("\n[Training]")
    history = {
        "trainLoss": [],
        "trainAccuracy": [],
        "validationLoss": [],
        "validationAccuracy": []
    }

    bestValidationLoss = float("inf")
    bestModelState = None

    startTime = time.time()

    for epoch in range(epochs):
        trainLoss, trainAccuracy = train(model, trainLoader, criterion, optimizer, device)
        validationLoss, validationAccuracy = validate(model, validationLoader, criterion, device)

        history["trainLoss"].append(trainLoss)
        history["trainAccuracy"].append(trainAccuracy)
        history["validationLoss"].append(validationLoss)
        history["validationAccuracy"].append(validationAccuracy)

        if validationLoss < bestValidationLoss:
            bestValidationLoss = validationLoss
            bestModelState = {key: value.cpu().clone() for key, value in model.state_dict().items()}

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {trainLoss:.4f} - Train Accuracy: {trainAccuracy:.4f} - Validation Loss: {validationLoss:.4f} - Validation Accuracy: {validationAccuracy:.4f}")

    trainingTime = time.time() - startTime

    model.load_state_dict(bestModelState)
    model.to(device)

    print(f"\nTraining time: {trainingTime:.2f} seconds")
    print(f"Best validation loss: {bestValidationLoss:.4f}")

    return history, trainingTime

def plotTrainingCurve(history):
    outputPath = Path("outputs")
    outputPath.mkdir(exist_ok=True)

    epochs = range(1, len(history["trainLoss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["trainLoss"], label="Train Loss")
    plt.plot(epochs, history["validationLoss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputPath / "training_loss.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(epochs, history["trainAccuracy"], label="Train Accuracy")
    plt.plot(epochs, history["validationAccuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputPath / "training_accuracy.png")
    plt.show()
    plt.close()

def evaluateTest(model, loader, criterion, device):
    print("\n[Test Evaluation]")
    model.eval()
    totalLoss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            totalLoss += loss.item() * images.size(0)

            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    testLoss = totalLoss / total
    testAccuracy = correct / total

    print(f"Test Loss: {testLoss:.4f}")
    print(f"Test Accuracy: {testAccuracy:.4f}")

    return testLoss, testAccuracy

def evaluateMetrics(model, loader, device):
    model.eval()
    allLabels = []
    allPredictions = []

    with torch.no_grad():

        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = (torch.sigmoid(outputs) >= 0.5).int().squeeze(1)

            allLabels.extend(labels.tolist())
            allPredictions.extend(predictions.cpu().tolist())

    confusionMatrix = confusion_matrix(allLabels, allPredictions)
    precision = precision_score(allLabels, allPredictions)
    recall = recall_score(allLabels, allPredictions)
    f1 = f1_score(allLabels, allPredictions)

    print("\n[Evaluation Metrics]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\n[Confusion Matrix]")
    print(confusionMatrix)

    return confusionMatrix, precision, recall, f1

def plotConfusionMatrix(confusionMatrix):
    outputPath = Path("outputs")
    outputPath.mkdir(exist_ok=True)

    plt.figure()
    plt.imshow(confusionMatrix)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Cat", "Dog"])
    plt.yticks([0, 1], ["Cat", "Dog"])

    for row in range(2):
        for column in range(2):
            plt.text(column, row, confusionMatrix[row, column], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outputPath / "confusion_matrix.png")
    plt.show()
    plt.close()
    
def getMisclassifiedImages(model, loader, device, limit=3):
    model.eval()
    misclassifiedImages = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = (torch.sigmoid(outputs) >= 0.5).int().squeeze(1)

            for index in range(len(labels)):
                if predictions[index].item() != labels[index].item():
                    misclassifiedImages.append((images[index].cpu(), labels[index].item(), predictions[index].item()))

                    if len(misclassifiedImages) == limit:
                        return misclassifiedImages

    return misclassifiedImages

def plotMisclassifiedImages(misclassifiedImages):
    outputPath = Path("outputs") / "misclassified"
    outputPath.mkdir(parents=True, exist_ok=True)

    for index, (image, trueLabel, predictedLabel) in enumerate(misclassifiedImages):
        image = image.permute(1, 2, 0)

        plt.figure()
        plt.imshow(image)
        plt.title(f"True: {'Cat' if trueLabel == 0 else 'Dog'} - Predicted: {'Cat' if predictedLabel == 0 else 'Dog'}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outputPath / f"example_{index + 1}.png")
        plt.show()
        plt.close()

def main():

    seed = 42
    batchSize = 32
    learningRate = 0.001
    epochs = 10

    setSeed(seed)

    datasetManager = DatasetManager()
    datasetManager.downloadDataset()
    datasetManager.inspectDataset()
    datasetManager.checkImageFiles()
    datasetManager.splitDataset()
    datasetManager.createTransforms()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasetManager.createDataLoaders()

    images, labels = next(iter(datasetManager.trainLoader))

    print("\n[DataLoader Check]")
    print(f"Image shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Pixel range: {images.min().item()} - {images.max().item()}")
    print(f"Labels: {labels[:10].tolist()}")

    model = SmallCNN().to(device)
    totalParameters = sum(parameter.numel() for parameter in model.parameters())
    print("\n[Model]")
    print(f"{model}")
    print(f"Total parameters: {totalParameters}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    print("\n[Training Configuration]")
    print(f"Random seed: {seed}")
    print(f"Device: {device}")
    print(f"Batch size: {batchSize}")
    print(f"Learning rate: {learningRate}")
    print(f"Epochs: {epochs}")
    print("Loss: BCEWithLogitsLoss")
    print("Optimizer: Adam")

    history, trainingTime = trainModel(model, datasetManager.trainLoader, datasetManager.validationLoader, criterion, optimizer, device, epochs)

    plotTrainingCurve(history)

    testLoss, testAccuracy = evaluateTest(model, datasetManager.testLoader, criterion, device)

    confusionMatrix, precision, recall, f1 = evaluateMetrics(model, datasetManager.testLoader, device)

    plotConfusionMatrix(confusionMatrix)

    misclassifiedImages = getMisclassifiedImages(model, datasetManager.testLoader, device)
    plotMisclassifiedImages(misclassifiedImages)

if __name__ == "__main__":
    main()