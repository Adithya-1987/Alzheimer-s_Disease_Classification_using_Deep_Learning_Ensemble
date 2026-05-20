"""
organize_dataset.py
===================
Organize the Alzheimer's dataset into train/val/test splits.
Splits: 70% train, 15% validation, 15% test
"""

import os
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

SOURCE_DIR = "akzimers dataset/AugmentedAlzheimerDataset"
TARGET_DIR = "dataset"

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def organize_dataset():
    """Split and organize dataset into train/val/test folders."""
    
    print("=" * 60)
    print("  Organizing Alzheimer's Dataset")
    print("=" * 60)
    
    # Create target directories
    for split in SPLITS.keys():
        for class_name in CLASSES:
            target_path = os.path.join(TARGET_DIR, split, class_name)
            os.makedirs(target_path, exist_ok=True)
    
    # Process each class
    for class_name in CLASSES:
        source_class_dir = os.path.join(SOURCE_DIR, class_name)
        
        if not os.path.exists(source_class_dir):
            print(f"[WARNING] {source_class_dir} not found, skipping...")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(source_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        total_images = len(image_files)
        print(f"\n[INFO] {class_name}: {total_images} images")
        
        # Shuffle for random split
        random.shuffle(image_files)
        
        # Calculate split indices
        train_end = int(total_images * SPLITS["train"])
        val_end = train_end + int(total_images * SPLITS["val"])
        
        splits_data = {
            "train": image_files[:train_end],
            "val": image_files[train_end:val_end],
            "test": image_files[val_end:]
        }
        
        # Copy files to respective splits
        for split_name, files in splits_data.items():
            print(f"  → {split_name}: {len(files)} images", end=" ")
            
            for filename in files:
                src = os.path.join(source_class_dir, filename)
                dst = os.path.join(TARGET_DIR, split_name, class_name, filename)
                shutil.copy2(src, dst)
            
            print("✓")
    
    print("\n" + "=" * 60)
    print("  Dataset organization complete!")
    print("=" * 60)
    
    # Print summary
    print("\nDataset Summary:")
    for split in SPLITS.keys():
        total = 0
        for class_name in CLASSES:
            class_dir = os.path.join(TARGET_DIR, split, class_name)
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total += count
        print(f"  {split:5s}: {total:6d} images")


if __name__ == "__main__":
    organize_dataset()
