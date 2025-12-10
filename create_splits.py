import os
import random
from collections import defaultdict

def create_train_val_test_splits(images_dir, annotations_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/validation/test splits for anemia classification dataset.
    
    Args:
        images_dir: Directory containing images
        annotations_dir: Directory containing XML annotations
        output_dir: Directory to save split files (ImageSets/Main)
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
    """
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    images = []
    
    for file in os.listdir(images_dir):
        if any(file.endswith(ext) for ext in image_extensions):
            base_name = os.path.splitext(file)[0]
            # Check if corresponding XML annotation exists
            xml_file = os.path.join(annotations_dir, base_name + '.xml')
            if os.path.exists(xml_file):
                images.append(base_name)
    
    print(f"Found {len(images)} images with corresponding annotations")
    
    # Group images by class for balanced splitting
    class_images = defaultdict(list)
    
    for image_name in images:
        # Extract class from filename
        if image_name.startswith('Anemic-') or image_name.startswith('Anmeic-'):
            class_images['anemic'].append(image_name)
        elif image_name.startswith('Non-anemic-') or image_name.startswith('Non-Anrmic-'):
            class_images['non_anemic'].append(image_name)
        elif image_name.startswith('Hand_'):
            class_images['hand_general'].append(image_name)
        else:
            print(f"Warning: Could not determine class for image: {image_name}")
            # Default to anemic if unclear
            class_images['anemic'].append(image_name)
    
    print(f"Class distribution:")
    for class_name, class_images_list in class_images.items():
        print(f"  {class_name}: {len(class_images_list)} images")
    
    # Create splits for each class
    train_images = []
    val_images = []
    test_images = []
    
    for class_name, class_images_list in class_images.items():
        # Shuffle images for this class
        random.shuffle(class_images_list)
        
        # Calculate split indices
        total = len(class_images_list)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split the data
        train_images.extend(class_images_list[:train_end])
        val_images.extend(class_images_list[train_end:val_end])
        test_images.extend(class_images_list[val_end:])
        
        print(f"{class_name} split: {len(class_images_list[:train_end])} train, "
              f"{len(class_images_list[train_end:val_end])} val, "
              f"{len(class_images_list[val_end:])} test")
    
    # Shuffle the final splits
    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)
    
    print(f"\nTotal split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write split files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for image_name in train_images:
            f.write(f"{image_name}\n")
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for image_name in val_images:
            f.write(f"{image_name}\n")
    
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for image_name in test_images:
            f.write(f"{image_name}\n")
    
    print(f"\nSplit files created in: {output_dir}")
    print("- train.txt")
    print("- val.txt") 
    print("- test.txt")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    images_dir = "dataset/Images"
    annotations_dir = "dataset/Annotations"
    output_dir = "dataset/ImageSets/Main"
    
    # Create splits
    create_train_val_test_splits(images_dir, annotations_dir, output_dir)