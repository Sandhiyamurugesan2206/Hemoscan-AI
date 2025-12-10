import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import logging
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATASET_DIR = 'dataset'
IMAGE_DIR = os.path.join(DATASET_DIR, 'Images')
ANNOTATION_DIR = os.path.join(DATASET_DIR, 'Annotations')
IMAGESETS_DIR = os.path.join(DATASET_DIR, 'ImageSets', 'Main')
OUTPUTS_DIR = 'outputs'
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')
LOGS_DIR = os.path.join(OUTPUTS_DIR, 'logs')
RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'results')

# Create output directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class mapping
CLASS_TO_IDX = {
    'anemic': 0,
    'non_anemic': 1,
    'hand_general': 2
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

def parse_xml_annotation(xml_file):
    """Parse Pascal VOC XML annotation file to extract class label."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find the object element and get the class name
        obj_element = root.find('object')
        if obj_element is not None:
            name_element = obj_element.find('name')
            if name_element is not None:
                class_name = name_element.text
                return CLASS_TO_IDX.get(class_name, 0)  # Default to anemic if unknown
        
        # Fallback: extract from filename
        filename = root.find('filename').text if root.find('filename') is not None else ""
        if 'anemic' in filename.lower() and 'non-anemic' not in filename.lower():
            return CLASS_TO_IDX['anemic']
        elif 'non-anemic' in filename.lower() or 'non-anrmic' in filename.lower():
            return CLASS_TO_IDX['non_anemic']
        elif 'hand' in filename.lower():
            return CLASS_TO_IDX['hand_general']
        else:
            return CLASS_TO_IDX['anemic']  # Default
            
    except Exception as e:
        logger.warning(f"Error parsing XML file {xml_file}: {e}")
        return CLASS_TO_IDX['anemic']  # Default to anemic

# Custom Dataset
class HemoglobinDataset(Dataset):
    def __init__(self, txt_file, image_dir, annotation_dir, transform=None):
        with open(txt_file, 'r') as f:
            self.image_ids = [line.strip() for line in f]
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        
        logger.info(f"Loaded {len(self.image_ids)} samples from {txt_file}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Try different image extensions
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            potential_path = os.path.join(self.image_dir, img_id + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"Image not found for ID: {img_id}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Parse XML annotation to get label
        xml_path = os.path.join(self.annotation_dir, img_id + '.xml')
        label = parse_xml_annotation(xml_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data transforms with medical image-appropriate augmentations
def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def evaluate_model(model, dataloader, device):
    """Evaluate model and return accuracy and predictions."""
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, all_predictions, all_labels

def train_model():
    """Main training function."""
    logger.info("Starting hemoglobin anemia classification training")
    
    # Datasets and Loaders
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
    
    test_dataset = HemoglobinDataset(
        txt_file=os.path.join(IMAGESETS_DIR, 'test.txt'),
        image_dir=IMAGE_DIR,
        annotation_dir=ANNOTATION_DIR,
        transform=get_transforms('val')
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training parameters
    num_epochs = 25
    best_val_acc = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    
    logger.info(f"Training for {num_epochs} epochs")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 40)
        
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
            
            if batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate training accuracy
        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        val_acc, val_predictions, val_labels = evaluate_model(model, val_loader, device)
        
        logger.info(f"Training Loss: {avg_loss:.4f}, Training Acc: {train_acc:.2f}%")
        logger.info(f"Validation Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'loss': avg_loss,
        }, checkpoint_path)
        
        scheduler.step()
    
    # Final evaluation on test set
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*50)
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    test_acc, test_predictions, test_labels = evaluate_model(model, test_loader, device)
    
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Create a simple classification report without sklearn
    class_names = [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]
    logger.info("\nPer-class results:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(test_labels) == i
        if class_mask.sum() > 0:
            class_correct = (np.array(test_predictions)[class_mask] == i).sum()
            class_total = class_mask.sum()
            class_acc = 100 * class_correct / class_total
            logger.info(f"{class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")
    
    logger.info(f"\nTraining completed! Best model saved at: {best_model_path}")
    return model

if __name__ == "__main__":
    train_model()