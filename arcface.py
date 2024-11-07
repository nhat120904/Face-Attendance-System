import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from dataset import FaceDataset
from torch.utils.tensorboard import SummaryWriter

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
writer = SummaryWriter(log_dir="runs/arcface")

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, feature_dim=512, margin=0.5, scale=30):
        super(ArcFaceModel, self).__init__()
        self.backbone = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.fc = nn.Identity()  # Remove classification head
        self.fc = nn.Linear(num_features, feature_dim)  # Project to embedding space
        self.margin = margin
        self.scale = scale
        self.arcface_loss = ArcFaceLoss(margin=self.margin, scale=self.scale, num_classes=num_classes, feature_dim=feature_dim)

    def forward(self, images, labels=None):
        embeddings = self.backbone(images)
        embeddings = self.fc(embeddings)
        
        if labels is not None:
            logits = self.arcface_loss(embeddings, labels)
            return logits, embeddings
        return embeddings

class ArcFaceLoss(nn.Module):
    def __init__(self, margin, scale, num_classes, feature_dim):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        normed_embeddings = nn.functional.normalize(embeddings)
        normed_weight = nn.functional.normalize(self.weight)
        cosine = torch.matmul(normed_embeddings, normed_weight.T)
        arccos = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        margin_cosine = torch.cos(arccos + self.margin)
        logits = self.scale * margin_cosine
        return logits
    
def train_arcface(model, train_loader, val_loader, num_epochs, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)  # Move model to GPU
    criterion = criterion.to(device)  # Move criterion to GPU
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU

            optimizer.zero_grad()
            logits, _ = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
    writer.close()

if __name__ == "__main__":
    
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FaceDataset('dataset', split='train', return_original=True, transform=transform)
    val_dataset = FaceDataset('dataset', split='val', return_original=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.classes)
    arcface_model = ArcFaceModel(num_classes=num_classes)
    train_arcface(arcface_model, train_loader, val_loader, num_epochs=20)

    # Save the model
    torch.save(arcface_model.state_dict(), 'arcface_model.pth')
