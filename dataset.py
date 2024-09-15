import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class FaceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, return_original=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.return_original = return_original
        
        if split == 'train':
            self.data_dir = os.path.join(root_dir, 'classification_data', 'train_data')
        elif split == 'val':
            self.data_dir = os.path.join(root_dir, 'classification_data', 'val_data')
        elif split == 'test':
            self.data_dir = os.path.join(root_dir, 'classification_data', 'test_data')
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")
        
        self.classes = os.listdir(self.data_dir)
        self.image_paths = self._get_image_paths()
        
        if split in ['val', 'test']:
            self.pairs = self._load_verification_pairs()

    def _get_image_paths(self):
        image_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_name))
        return image_paths

    def _load_verification_pairs(self):
        pairs_file = os.path.join(self.root_dir, 'verification_pairs_val.txt')
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                img1_path, img2_path, label = line.strip().split()
                pairs.append((
                    os.path.join(self.root_dir, img1_path),
                    os.path.join(self.root_dir, img2_path),
                    int(label)
                ))
        return pairs

    def __len__(self):
        if self.split == 'train':
            return len(self.image_paths)
        else:
            return len(self.pairs)

    def __getitem__(self, idx):
        if self.split == 'train':
            anchor_path = self.image_paths[idx]
            anchor_class = os.path.basename(os.path.dirname(anchor_path))
            
            # Select positive (same class) and negative (different class) samples
            positive_path = random.choice([p for p in self.image_paths if os.path.basename(os.path.dirname(p)) == anchor_class and p != anchor_path])
            negative_path = random.choice([p for p in self.image_paths if os.path.basename(os.path.dirname(p)) != anchor_class])
            
            anchor = Image.open(anchor_path).convert('RGB')
            positive = Image.open(positive_path).convert('RGB')
            negative = Image.open(negative_path).convert('RGB')
            
            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)
            
            return anchor, positive, negative
        else:
            img1_path, img2_path, label = self.pairs[idx]
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            return img1, img2, torch.tensor(label, dtype=torch.float32)

def visualize_pairs(dataset, num_pairs=5):
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 4*num_pairs))
    for i in range(num_pairs):
        img1, img2, label = dataset[random.randint(0, len(dataset)-1)]
        axes[i, 0].imshow(img1)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(img2)
        axes[i, 1].axis('off')
        axes[i, 0].set_title(f"Pair {i+1}")
        axes[i, 1].set_title(f"Same: {bool(label)}")
    plt.tight_layout()
    plt.show()
