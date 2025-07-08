"""
YOLO v8 Character-Level Detection Training Pipeline
Trains a custom YOLO v8 model for individual character detection in serial numbers
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path
import yaml

class YOLOv8CharacterTrainer:
    """
    Train YOLO v8 for character-level detection
    """
    
    def __init__(self, data_dir="character_annotations_output"):
        self.data_dir = Path(data_dir)
        self.classes = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
            'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 
            'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
            'W': 22, 'X': 23, 'Y': 24, 'Z': 25, '0': 26, '1': 27, '2': 28,
            '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35
        }
        
        self.setup_yolo_dataset()
    
    def setup_yolo_dataset(self):
        """Convert your annotations to YOLO v8 format"""
        
        # Create YOLO dataset structure
        dataset_dir = Path("yolo_character_dataset")
        dataset_dir.mkdir(exist_ok=True)
        
        # Create train/val directories
        for split in ['train', 'val']:
            (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {v: k for k, v in self.classes.items()}
        }
        
        with open(dataset_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)
        
        print(f"✅ YOLO v8 dataset structure created at {dataset_dir}")
        return dataset_dir
    
    def convert_annotations_to_yolo(self):
        """Convert your character annotations to YOLO format"""
        
        json_dir = self.data_dir / "json_annotations"
        dataset_dir = Path("yolo_character_dataset")
        
        train_count = 0
        val_count = 0
        
        for json_file in json_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            image_filename = data['image_info']['filename']
            image_width = data['image_info']['width']
            image_height = data['image_info']['height']
            
            # Split 80% train, 20% val
            is_train = np.random.random() < 0.8
            split = 'train' if is_train else 'val'
            
            # Copy image
            src_image = Path(image_filename)  # Adjust path as needed
            dst_image = dataset_dir / split / 'images' / image_filename
            
            # Create YOLO label file
            label_file = dataset_dir / split / 'labels' / f"{src_image.stem}.txt"
            
            with open(label_file, 'w') as f:
                # Write character-level annotations
                for char_data in data.get('character_annotations', []):
                    char = char_data['character']
                    if char in self.classes:
                        class_id = self.classes[char]
                        
                        # Convert to YOLO format (normalized)
                        x_min, y_min, x_max, y_max = char_data['bbox']
                        
                        center_x = (x_min + x_max) / 2 / image_width
                        center_y = (y_min + y_max) / 2 / image_height
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height
                        
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            if is_train:
                train_count += 1
            else:
                val_count += 1
        
        print(f"✅ Converted {train_count} training images, {val_count} validation images")
    
    def train_yolo_model(self):
        """Train YOLO v8 model for character detection"""
        
        # Initialize YOLO v8 model
        model = YOLO('yolov8n.pt')  # Start with nano model for speed
        
        # Train the model
        results = model.train(
            data='yolo_character_dataset/dataset.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            name='character_detection',
            patience=10,
            save_period=10,
            device='0' if torch.cuda.is_available() else 'cpu'
        )
        
        print("✅ YOLO v8 character detection model trained!")
        return model
    
    def test_trained_model(self, model_path, test_image_path):
        """Test the trained YOLO model"""
        
        # Load trained model
        model = YOLO(model_path)
        
        # Run inference
        results = model(test_image_path)
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get character name
                    char_name = list(self.classes.keys())[list(self.classes.values()).index(class_id)]
                    
                    print(f"Character: {char_name}, Confidence: {confidence:.2f}, Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        return results

def main():
    """Main training pipeline"""
    
    print("🚀 Starting YOLO v8 Character Detection Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = YOLOv8CharacterTrainer()
    
    # Convert annotations
    print("📝 Converting annotations to YOLO format...")
    trainer.convert_annotations_to_yolo()
    
    # Train model
    print("🎯 Training YOLO v8 model...")
    model = trainer.train_yolo_model()
    
    print("✅ Training complete!")
    print("📁 Model saved in: runs/detect/character_detection/")
    print("🎯 Use the best.pt file for inference")

if __name__ == "__main__":
    main()
