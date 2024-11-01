# train.py
# Author: Nguyen Cuong Nhat
# COS30082 - Applied Machine Learning 
# Project: Face Attendance System

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import FaceEmbeddingCNN
from dataset import FaceDataset
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    print(f"Using device: {device}")
    criterion = nn.TripletMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in train_loader:
            # Print shapes of input tensors
            print(f"Anchor shape: {anchor.shape}")
            print(f"Positive shape: {positive.shape}")
            print(f"Negative shape: {negative.shape}")
            
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch+1)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    writer.close()
    return model

def evaluate_model(model, data_loader, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in data_loader:
            img1, img2 = img1.to(device), img2.to(device)
            
            embedding1 = model(img1)
            embedding2 = model(img2)
            
            similarity = compute_similarity(embedding1, embedding2)
            predictions = (similarity > threshold).float()
            
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.size(0)
            
            all_similarities.extend(similarity.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = correct / total
    auc_score = roc_auc_score(all_labels, all_similarities)
    
    return accuracy, auc_score

def compute_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2)

def find_best_threshold(model, val_loader):
    model.eval()
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2 = img1.to(device), img2.to(device)
            
            embedding1 = model(img1)
            embedding2 = model(img2)
            
            similarity = compute_similarity(embedding1, embedding2)
            
            all_similarities.extend(similarity.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    thresholds = np.arange(0, 1, 0.01)
    best_accuracy = 0
    best_threshold = 0
    
    for threshold in thresholds:
        predictions = (np.array(all_similarities) > threshold).astype(int)
        accuracy = (predictions == np.array(all_labels)).mean()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FaceDataset('dataset', split='train')
    val_dataset = FaceDataset('dataset', split='val')
    test_dataset = FaceDataset('dataset', split='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model
    model = FaceEmbeddingCNN().to(device)

    # Train model
    model = train_model(model, train_loader, num_epochs=10)

    # Find the best threshold using validation set
    best_threshold = find_best_threshold(model, val_loader)
    print(f"Best threshold: {best_threshold:.4f}")

    # Evaluate on validation set
    val_accuracy, val_auc = evaluate_model(model, val_loader, threshold=best_threshold)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Evaluate on test set
    test_accuracy, test_auc = evaluate_model(model, test_loader, threshold=best_threshold)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'face_embedding_model_v1.pth')