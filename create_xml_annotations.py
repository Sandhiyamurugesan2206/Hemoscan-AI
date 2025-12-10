#!/usr/bin/env python3
"""
Script to automatically generate XML annotation files for hemoglobin anemia classification dataset.
Extracts class labels from image filenames and creates Pascal VOC format XML annotations.
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
import glob

def get_label_from_filename(filename):
    """
    Extract class label from filename.
    Returns 'anemic' or 'non_anemic' based on filename pattern.
    """
    filename_lower = filename.lower()
    
    # Check for anemic patterns
    if 'anemic-fn' in filename_lower or 'anmeic-fn' in filename_lower:
        return 'anemic'
    # Check for non-anemic patterns
    elif 'non-anemic' in filename_lower or 'non-anrmic' in filename_lower:
        return 'non_anemic'
    # Check for hand images (assuming they are general hand/palm images)
    elif 'hand_' in filename_lower:
        # You may need to manually classify these or use a different strategy
        # For now, treating them as a separate category
        return 'hand_general'
    else:
        # Default fallback
        return 'unknown'

def create_xml_annotation(image_path, annotation_dir):
    """
    Create a Pascal VOC format XML annotation file for an image.
    """
    try:
        # Get image info
        with Image.open(image_path) as img:
            width, height = img.size
            depth = len(img.getbands()) if img.mode != 'L' else 1
        
        filename = os.path.basename(image_path)
        image_name = os.path.splitext(filename)[0]
        
        # Get class label from filename
        class_label = get_label_from_filename(filename)
        
        # Create XML structure
        root = ET.Element("annotation")
        
        # Folder
        folder = ET.SubElement(root, "folder")
        folder.text = "Images"
        
        # Filename
        filename_elem = ET.SubElement(root, "filename")
        filename_elem.text = filename
        
        # Path
        path = ET.SubElement(root, "path")
        path.text = image_path
        
        # Source
        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Hemoglobin Anemia Dataset"
        
        # Size
        size = ET.SubElement(root, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(height)
        depth_elem = ET.SubElement(size, "depth")
        depth_elem.text = str(depth)
        
        # Segmented
        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"
        
        # Object (for image classification, we create one object covering the whole image)
        obj = ET.SubElement(root, "object")
        
        # Name (class label)
        name = ET.SubElement(obj, "name")
        name.text = class_label
        
        # Pose
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        # Truncated
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        # Difficult
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        # Bounding box (whole image for classification)
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = "1"
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = "1"
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(width)
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(height)
        
        # Create pretty XML
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="    ")
        
        # Remove empty lines
        pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
        
        # Save XML file
        xml_filename = image_name + '.xml'
        xml_path = os.path.join(annotation_dir, xml_filename)
        
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        return xml_path, class_label
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

def main():
    # Paths
    dataset_dir = 'dataset'
    images_dir = os.path.join(dataset_dir, 'Images')
    annotations_dir = os.path.join(dataset_dir, 'Annotations')
    
    # Create annotations directory if it doesn't exist
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images")
    
    # Statistics
    class_counts = {}
    created_count = 0
    
    # Process each image
    for image_path in image_files:
        xml_path, class_label = create_xml_annotation(image_path, annotations_dir)
        
        if xml_path:
            created_count += 1
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
            
            if created_count % 100 == 0:
                print(f"Processed {created_count} images...")
    
    print(f"\nCreated {created_count} XML annotation files")
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    print(f"\nXML files saved in: {annotations_dir}")

if __name__ == "__main__":
    main()