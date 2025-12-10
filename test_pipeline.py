import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import logging
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from main training script
from hemoglobin_train import HemoglobinDataset, get_transforms, CLASS_TO_IDX, IDX_TO_CLASS

def quick_test():
    """Quick test of the training pipeline with minimal epochs."""
    logger.info("Starting quick test of hemoglobin training pipeline")
    
    # Paths
    IMAGESETS_DIR = os.path.join('dataset', 'ImageSets', 'Main')
    IMAGE_DIR = os.path.join('dataset', 'Images')
    ANNOTATION_DIR = os.path.join('dataset', 'Annotations')
    
    # Create small test datasets (first 100 samples only)
    train_dataset = HemoglobinDataset(
        txt_file=os.path.join(IMAGESETS_DIR, 'train.txt'),
        image_dir=IMAGE_DIR,
        annotation_dir=ANNOTATION_DIR,
        transform=get_transforms('train')
    )
    
    val_dataset = HemoglobinDataset(
        txt_file=os.path.join(IMAGESETS_DIR, 'val.txt'),
        image_dir=IMAGE_DIR,
        annotation_dir=ANNOTATION_DIR,
        transform=get_transforms('val')
    )
    
    # Use subset for quick test
    train_subset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
    val_subset = torch.utils.data.Subset(val_dataset, range(min(50, len(val_dataset))))
    
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=0)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load pretrained ResNet18 and modify for 3 classes
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_TO_IDX))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Quick training test (just 2 epochs)
    num_epochs = 2
    logger.info(f"Quick test with {num_epochs} epochs")
    logger.info(f"Train samples: {len(train_subset)}")
    logger.info(f"Validation samples: {len(val_subset)}")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            logger.info(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate training accuracy
        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        
        logger.info(f"Training Loss: {avg_loss:.4f}, Training Acc: {train_acc:.2f}%")
        logger.info(f"Validation Acc: {val_acc:.2f}%")
    
    logger.info("\n" + "="*50)
    logger.info("QUICK TEST COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info("The training pipeline is working correctly.")
    logger.info("You can now run the full training with: python hemoglobin_train.py")
    
    return True

if __name__ == "__main__":
    quick_test()