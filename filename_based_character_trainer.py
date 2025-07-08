"""
Improved YOLO v8 Character-Level Detection Training Pipeline
Uses filename-based labels from test_images folder where filename = serial number
"""

import cv2
import numpy as np
import json
from pathlib import Path
import yaml
import re
import shutil
from typing import Dict, List, Tuple
import easyocr
import torch

class FilenameBasedCharacterTrainer:
    """
    Train YOLO v8 for character-level detection using filename-based labels
    """
    
    def __init__(self, test_images_dir="test_images"):
        self.test_images_dir = Path(test_images_dir)
        self.classes = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
            'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 
            'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
            'W': 22, 'X': 23, 'Y': 24, 'Z': 25, '0': 26, '1': 27, '2': 28,
            '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35
        }
        
        # Initialize EasyOCR for text detection
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Serial number patterns
        self.serial_patterns = [
            r'^[A-Z]{2}\d{4}[A-Z]$',  # AAnnnnA (FD8764S)
            r'^[A-Z]{2}\d{3}[A-Z]\d$', # AAnnnAn (GFD887E) 
            r'^[A-Z]{3}\d{3}[A-Z]$'   # AAAnnnA
        ]
        
        print(f"✅ Initialized trainer with {len(list(self.test_images_dir.glob('*.jpg')))} images")
    
    def validate_serial_number(self, serial: str) -> bool:
        """Check if serial number matches expected patterns"""
        return any(re.match(pattern, serial) for pattern in self.serial_patterns)
    
    def detect_text_region(self, image_path: Path) -> Dict:
        """Detect the text region in the image"""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use EasyOCR to detect text regions
        results = self.reader.readtext(image_rgb, detail=1)
        
        if not results:
            return None
        
        # Find the most confident detection
        best_detection = max(results, key=lambda x: x[2])
        bbox, text, confidence = best_detection
        
        # Convert bbox to [x_min, y_min, x_max, y_max]
        bbox_np = np.array(bbox, dtype=np.int32)
        x_min, y_min = bbox_np[:, 0].min(), bbox_np[:, 1].min()
        x_max, y_max = bbox_np[:, 0].max(), bbox_np[:, 1].max()
        
        return {
            'bbox': [x_min, y_min, x_max, y_max],
            'text': text.strip(),
            'confidence': confidence
        }
    
    def segment_characters(self, image_path: Path, text_region: Dict, expected_serial: str) -> List[Dict]:
        """Segment individual characters from the text region"""
        image = cv2.imread(str(image_path))
        x_min, y_min, x_max, y_max = text_region['bbox']
        
        # Crop the text region
        text_crop = image[y_min:y_max, x_min:x_max]
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by x-coordinate (left to right)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        characters = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small contours
            if w < 10 or h < 10:
                continue
                
            # Convert back to original image coordinates
            char_x_min = x_min + x
            char_y_min = y_min + y
            char_x_max = x_min + x + w
            char_y_max = y_min + y + h
            
            # Get expected character if available
            expected_char = expected_serial[i] if i < len(expected_serial) else 'X'
            
            characters.append({
                'bbox': [char_x_min, char_y_min, char_x_max, char_y_max],
                'character': expected_char,
                'contour_area': cv2.contourArea(contour)
            })
        
        # If we have fewer characters than expected, use uniform segmentation
        if len(characters) < len(expected_serial):
            return self.uniform_character_segmentation(text_region, expected_serial)
        
        return characters[:len(expected_serial)]  # Take only expected number of characters
    
    def uniform_character_segmentation(self, text_region: Dict, expected_serial: str) -> List[Dict]:
        """Fallback: uniform segmentation when contour detection fails"""
        x_min, y_min, x_max, y_max = text_region['bbox']
        
        width = x_max - x_min
        height = y_max - y_min
        char_width = width / len(expected_serial)
        
        characters = []
        for i, char in enumerate(expected_serial):
            char_x_min = int(x_min + i * char_width)
            char_y_min = y_min
            char_x_max = int(x_min + (i + 1) * char_width)
            char_y_max = y_max
            
            characters.append({
                'bbox': [char_x_min, char_y_min, char_x_max, char_y_max],
                'character': char,
                'method': 'uniform'
            })
        
        return characters
    
    def process_image_file(self, image_path: Path) -> Dict:
        """Process a single image file and create character-level annotations"""
        # Extract serial number from filename
        serial_number = image_path.stem.upper()
        
        # Validate serial number
        if not self.validate_serial_number(serial_number):
            print(f"⚠️  Invalid serial number pattern: {serial_number}")
            return None
        
        # Detect text region
        text_region = self.detect_text_region(image_path)
        if not text_region:
            print(f"❌ No text detected in {image_path.name}")
            return None
        
        # Get image dimensions
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        
        # Segment characters
        characters = self.segment_characters(image_path, text_region, serial_number)
        
        return {
            'image_path': image_path,
            'serial_number': serial_number,
            'image_size': (width, height),
            'text_region': text_region,
            'characters': characters
        }
    
    def create_yolo_dataset(self, output_dir="yolo_character_dataset"):
        """Create YOLO v8 dataset from processed images"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create train/val directories
        for split in ['train', 'val']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {v: k for k, v in self.classes.items()}
        }
        
        with open(output_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)
        
        # Process all images
        train_count = 0
        val_count = 0
        successful_images = 0
        
        image_files = list(self.test_images_dir.glob('*.jpg')) + list(self.test_images_dir.glob('*.png'))
        
        for image_file in image_files:
            print(f"Processing {image_file.name}...")
            
            result = self.process_image_file(image_file)
            if not result:
                continue
            
            # Split 80% train, 20% val
            is_train = np.random.random() < 0.8
            split = 'train' if is_train else 'val'
            
            # Copy image
            dst_image = output_dir / split / 'images' / image_file.name
            shutil.copy2(image_file, dst_image)
            
            # Create label file
            label_file = output_dir / split / 'labels' / f"{image_file.stem}.txt"
            
            with open(label_file, 'w') as f:
                for char_data in result['characters']:
                    char = char_data['character']
                    if char in self.classes:
                        class_id = self.classes[char]
                        
                        # Convert to YOLO format (normalized)
                        x_min, y_min, x_max, y_max = char_data['bbox']
                        
                        center_x = (x_min + x_max) / 2 / result['image_size'][0]
                        center_y = (y_min + y_max) / 2 / result['image_size'][1]
                        width = (x_max - x_min) / result['image_size'][0]
                        height = (y_max - y_min) / result['image_size'][1]
                        
                        # Ensure values are within [0, 1]
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            successful_images += 1
            if is_train:
                train_count += 1
            else:
                val_count += 1
        
        print(f"\n✅ YOLO Dataset Created Successfully!")
        print(f"📊 Total processed: {successful_images} images")
        print(f"🎯 Training images: {train_count}")
        print(f"🎯 Validation images: {val_count}")
        print(f"📁 Dataset saved to: {output_dir}")
        
        return output_dir
    
    def create_visualization(self, output_dir="character_visualization"):
        """Create visualization of character segmentation"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Process first 10 images for visualization
        image_files = list(self.test_images_dir.glob('*.jpg'))[:10]
        
        for image_file in image_files:
            result = self.process_image_file(image_file)
            if not result:
                continue
            
            # Load image
            image = cv2.imread(str(image_file))
            
            # Draw text region
            text_region = result['text_region']
            x_min, y_min, x_max, y_max = text_region['bbox']
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw character boxes
            for i, char_data in enumerate(result['characters']):
                x_min, y_min, x_max, y_max = char_data['bbox']
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                # Add character label
                cv2.putText(image, char_data['character'], (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add serial number title
            cv2.putText(image, f"Serial: {result['serial_number']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save visualization
            output_path = output_dir / f"{image_file.stem}_annotated.jpg"
            cv2.imwrite(str(output_path), image)
        
        print(f"✅ Visualizations saved to: {output_dir}")

def main():
    """Main function to create YOLO dataset from filename-labeled images"""
    
    print("🚀 Creating YOLO v8 Dataset from Filename-Labeled Images")
    print("=" * 70)
    
    # Initialize trainer
    trainer = FilenameBasedCharacterTrainer()
    
    # Create visualizations first
    print("📸 Creating character segmentation visualizations...")
    trainer.create_visualization()
    
    # Create YOLO dataset
    print("\n📝 Creating YOLO v8 dataset...")
    dataset_dir = trainer.create_yolo_dataset()
    
    print("\n🎯 Next Steps:")
    print("1. Install ultralytics: pip install ultralytics")
    print("2. Train YOLO model:")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('yolov8n.pt')")
    print(f"   model.train(data='{dataset_dir}/dataset.yaml', epochs=100)")
    print("3. Check visualizations in 'character_visualization' folder")

if __name__ == "__main__":
    main()
