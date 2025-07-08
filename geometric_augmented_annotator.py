import torch
import cv2
import numpy as np
import easyocr
import re
import json
import os
from pathlib import Path
import time
from datetime import datetime
import argparse
import math
from typing import List, Tuple, Dict, Any

class GeometricAugmentedAnnotator:
    """
    Enhanced Serial Number Annotator with Geometric Augmentation
    - Rotates images and properly adjusts bounding boxes
    - Maintains character-level and word-level annotations
    - Generates augmented training data for complex scenarios
    """
    
    def __init__(self, output_dir="augmented_annotations_output"):
        print("🚀 Initializing Geometric Augmented Annotator...")
        
        # Create output directory structure
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.json_dir = self.output_dir / "json_annotations"
        self.yolo_dir = self.output_dir / "yolo_annotations"
        self.visual_dir = self.output_dir / "visual_annotations"
        self.augmented_images_dir = self.output_dir / "augmented_images"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.json_dir, self.yolo_dir, self.visual_dir, self.augmented_images_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"🎯 CUDA detected! GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, using CPU")
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=cuda_available)
        
        # Define serial number patterns
        self.serial_patterns = [
            r'^[A-Z]{2}\d{4}[A-Z]$',  # AAnnnnA (FD8764S)
            r'^[A-Z]{2}\d{3}[A-Z]\d$', # AAnnnAn (GFD887E) 
            r'^[A-Z]{3}\d{3}[A-Z]$'   # AAAnnnA
        ]
        
        self.pattern_names = ['AAnnnnA', 'AAnnnAn', 'AAAnnnA']
        
        # Character mapping for training
        self.char_to_id = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
            'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 
            'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
            'W': 22, 'X': 23, 'Y': 24, 'Z': 25, '0': 26, '1': 27, '2': 28,
            '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35
        }
        
        # Augmentation parameters
        self.rotation_angles = [-15, -10, -5, 0, 5, 10, 15]  # degrees
        self.brightness_factors = [0.7, 0.85, 1.0, 1.15, 1.3]
        self.contrast_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'total_augmented': 0,
            'images_with_serials': 0,
            'total_serials_found': 0,
            'character_count': {char: 0 for char in self.char_to_id.keys()},
            'processing_times': [],
            'errors': []
        }
        
        print("✅ Geometric Augmented Annotator initialized successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
    
    def rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate image by given angle and return rotation matrix
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        
        return rotated_image, rotation_matrix
    
    def rotate_bounding_box(self, bbox: List[int], rotation_matrix: np.ndarray) -> List[int]:
        """
        Rotate bounding box coordinates using rotation matrix
        """
        x1, y1, x2, y2 = bbox
        
        # Create corner points
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ]).T
        
        # Apply rotation
        rotated_corners = rotation_matrix @ corners
        
        # Get new bounding box
        x_coords = rotated_corners[0, :]
        y_coords = rotated_corners[1, :]
        
        new_x1 = int(np.min(x_coords))
        new_y1 = int(np.min(y_coords))
        new_x2 = int(np.max(x_coords))
        new_y2 = int(np.max(y_coords))
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def rotate_polygon(self, polygon: List[List[int]], rotation_matrix: np.ndarray) -> List[List[int]]:
        """
        Rotate polygon coordinates using rotation matrix
        """
        rotated_polygon = []
        
        for point in polygon:
            # Convert to homogeneous coordinates
            point_homo = np.array([point[0], point[1], 1])
            
            # Apply rotation
            rotated_point = rotation_matrix @ point_homo
            
            rotated_polygon.append([int(rotated_point[0]), int(rotated_point[1])])
        
        return rotated_polygon
    
    def adjust_brightness_contrast(self, image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
        """
        Adjust image brightness and contrast
        """
        # Convert to float for calculations
        image_float = image.astype(np.float32)
        
        # Apply brightness and contrast
        adjusted = image_float * contrast + brightness * 255
        
        # Clip values and convert back to uint8
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def detect_text_and_characters(self, image: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect text and segment into characters
        """
        # Detect text regions
        results = self.reader.readtext(image, detail=1, paragraph=False)
        
        all_detections = []
        character_detections = []
        
        for bbox, text, confidence in results:
            if confidence > 0.1:
                bbox_np = np.array(bbox, dtype=np.int32)
                x_min, y_min = bbox_np[:, 0].min(), bbox_np[:, 1].min()
                x_max, y_max = bbox_np[:, 0].max(), bbox_np[:, 1].max()
                
                # Convert bbox to list if it's not already
                if isinstance(bbox, np.ndarray):
                    bbox_poly = bbox.tolist()
                else:
                    bbox_poly = bbox
                
                word_detection = {
                    'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                    'bbox_poly': bbox_poly,
                    'text': text.strip(),
                    'confidence': float(confidence)
                }
                
                all_detections.append(word_detection)
                
                # Character-level segmentation
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean_text) > 0:
                    char_width = (x_max - x_min) / len(clean_text)
                    
                    for i, char in enumerate(clean_text):
                        if char in self.char_to_id:
                            char_x1 = int(x_min + i * char_width)
                            char_x2 = int(x_min + (i + 1) * char_width)
                            
                            character_detections.append({
                                'bbox': [char_x1, y_min, char_x2, y_max],
                                'character': char,
                                'class_id': self.char_to_id[char],
                                'confidence': float(confidence),
                                'parent_word': text.strip()
                            })
        
        return all_detections, character_detections
    
    def filter_serial_numbers(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter detections to find valid serial numbers
        """
        serial_numbers = []
        
        for detection in detections:
            text = detection['text'].upper()
            clean_text = re.sub(r'[^A-Z0-9]', '', text)
            
            # Check different lengths for flexibility
            for length in [6, 7, 8]:
                if len(clean_text) == length:
                    if length == 6:
                        test_texts = [clean_text + 'X', 'X' + clean_text]
                    elif length == 8:
                        test_texts = [clean_text[1:], clean_text[:-1]]
                    else:
                        test_texts = [clean_text]
                    
                    for test_text in test_texts:
                        if len(test_text) == 7:
                            for i, pattern in enumerate(self.serial_patterns):
                                if re.match(pattern, test_text):
                                    detection_copy = detection.copy()
                                    detection_copy.update({
                                        'serial_number': test_text,
                                        'pattern': self.pattern_names[i],
                                        'pattern_index': i,
                                        'original_text': text
                                    })
                                    serial_numbers.append(detection_copy)
                                    break
                            if 'serial_number' in detection:
                                break
        
        return serial_numbers
    
    def create_augmented_version(self, image: np.ndarray, all_detections: List[Dict], 
                               character_detections: List[Dict], serial_numbers: List[Dict],
                               angle: float, brightness: float, contrast: float) -> Dict:
        """
        Create augmented version with rotated image and adjusted annotations
        """
        # Apply rotation
        rotated_image, rotation_matrix = self.rotate_image(image, angle)
        
        # Apply brightness/contrast adjustment
        final_image = self.adjust_brightness_contrast(rotated_image, brightness, contrast)
        
        # Rotate all bounding boxes
        rotated_all_detections = []
        for detection in all_detections:
            rotated_detection = detection.copy()
            rotated_detection['bbox'] = self.rotate_bounding_box(detection['bbox'], rotation_matrix)
            rotated_detection['bbox_poly'] = self.rotate_polygon(detection['bbox_poly'], rotation_matrix)
            rotated_all_detections.append(rotated_detection)
        
        # Rotate character bounding boxes
        rotated_character_detections = []
        for char_detection in character_detections:
            rotated_char = char_detection.copy()
            rotated_char['bbox'] = self.rotate_bounding_box(char_detection['bbox'], rotation_matrix)
            rotated_character_detections.append(rotated_char)
        
        # Rotate serial number bounding boxes
        rotated_serial_numbers = []
        for serial in serial_numbers:
            rotated_serial = serial.copy()
            rotated_serial['bbox'] = self.rotate_bounding_box(serial['bbox'], rotation_matrix)
            rotated_serial['bbox_poly'] = self.rotate_polygon(serial['bbox_poly'], rotation_matrix)
            rotated_serial_numbers.append(rotated_serial)
        
        return {
            'image': final_image,
            'all_detections': rotated_all_detections,
            'character_detections': rotated_character_detections,
            'serial_numbers': rotated_serial_numbers,
            'augmentation_params': {
                'rotation_angle': angle,
                'brightness_factor': brightness,
                'contrast_factor': contrast
            }
        }
    
    def save_augmented_annotations(self, image_path: Path, original_size: Tuple[int, int], 
                                 augmented_data: Dict, aug_suffix: str) -> Dict:
        """
        Save all annotation formats for augmented data
        """
        base_name = f"{image_path.stem}_{aug_suffix}"
        
        # Save augmented image
        image_save_path = self.augmented_images_dir / f"{base_name}.jpg"
        cv2.imwrite(str(image_save_path), cv2.cvtColor(augmented_data['image'], cv2.COLOR_RGB2BGR))
        
        # Save JSON annotation
        annotation = {
            'image_info': {
                'filename': f"{base_name}.jpg",
                'original_filename': image_path.name,
                'width': augmented_data['image'].shape[1],
                'height': augmented_data['image'].shape[0],
                'channels': 3
            },
            'augmentation_info': augmented_data['augmentation_params'],
            'detection_summary': {
                'total_text_detections': len(augmented_data['all_detections']),
                'character_detections': len(augmented_data['character_detections']),
                'serial_numbers_found': len(augmented_data['serial_numbers']),
                'patterns_detected': list(set([sn['pattern'] for sn in augmented_data['serial_numbers']]))
            },
            'all_text_detections': augmented_data['all_detections'],
            'character_annotations': augmented_data['character_detections'],
            'serial_numbers': augmented_data['serial_numbers'],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'detector_version': '2.0_augmented',
                'patterns_supported': self.pattern_names
            }
        }
        
        json_path = self.json_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Save YOLO format for characters
        yolo_char_path = self.yolo_dir / f"{base_name}_characters.txt"
        with open(yolo_char_path, 'w') as f:
            for char_data in augmented_data['character_detections']:
                x1, y1, x2, y2 = char_data['bbox']
                
                # Normalize coordinates
                center_x = (x1 + x2) / 2 / augmented_data['image'].shape[1]
                center_y = (y1 + y2) / 2 / augmented_data['image'].shape[0]
                width = (x2 - x1) / augmented_data['image'].shape[1]
                height = (y2 - y1) / augmented_data['image'].shape[0]
                
                class_id = char_data['class_id']
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Save YOLO format for words
        yolo_word_path = self.yolo_dir / f"{base_name}_words.txt"
        with open(yolo_word_path, 'w') as f:
            for serial in augmented_data['serial_numbers']:
                x1, y1, x2, y2 = serial['bbox']
                
                # Normalize coordinates
                center_x = (x1 + x2) / 2 / augmented_data['image'].shape[1]
                center_y = (y1 + y2) / 2 / augmented_data['image'].shape[0]
                width = (x2 - x1) / augmented_data['image'].shape[1]
                height = (y2 - y1) / augmented_data['image'].shape[0]
                
                # Class ID 0 for serial numbers
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Save visual annotation
        visual_path = self.save_visual_annotation(base_name, augmented_data)
        
        return {
            'image_path': str(image_save_path),
            'json_path': str(json_path),
            'yolo_char_path': str(yolo_char_path),
            'yolo_word_path': str(yolo_word_path),
            'visual_path': str(visual_path)
        }
    
    def save_visual_annotation(self, base_name: str, augmented_data: Dict) -> Path:
        """
        Save visual annotation with bounding boxes
        """
        result_image = augmented_data['image'].copy()
        
        # Draw character bounding boxes in green
        for char_data in augmented_data['character_detections']:
            x1, y1, x2, y2 = char_data['bbox']
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add character label
            label = f"{char_data['character']}"
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw serial number bounding boxes in red
        for serial in augmented_data['serial_numbers']:
            x1, y1, x2, y2 = serial['bbox']
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Add serial number label
            label = f"SERIAL: {serial['serial_number']}"
            label_pos = (x1, max(y1 - 40, 30))
            
            # Background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_image,
                         (label_pos[0] - 5, label_pos[1] - text_height - 5),
                         (label_pos[0] + text_width + 5, label_pos[1] + baseline + 5),
                         (255, 0, 0), -1)
            
            cv2.putText(result_image, label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add augmentation info
        aug_info = f"Rot: {augmented_data['augmentation_params']['rotation_angle']}°"
        cv2.putText(result_image, aug_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save image
        visual_path = self.visual_dir / f"{base_name}_annotated.jpg"
        cv2.imwrite(str(visual_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        return visual_path
    
    def process_image_with_augmentation(self, image_path: Path) -> List[Dict]:
        """
        Process a single image with multiple augmentations
        """
        print(f"📸 Processing with augmentation: {image_path.name}")
        
        start_time = time.time()
        results = []
        
        try:
            # Load and preprocess original image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = (image_rgb.shape[1], image_rgb.shape[0])
            
            # Detect text and characters in original image
            all_detections, character_detections = self.detect_text_and_characters(image_rgb)
            serial_numbers = self.filter_serial_numbers(all_detections)
            
            if not serial_numbers:
                print(f"   ⚠️  No serial numbers found in {image_path.name}")
                return results
            
            # Generate augmented versions
            aug_count = 0
            for angle in self.rotation_angles:
                for brightness in self.brightness_factors:
                    for contrast in self.contrast_factors:
                        # Skip original (no augmentation)
                        if angle == 0 and brightness == 1.0 and contrast == 1.0:
                            continue
                        
                        aug_suffix = f"rot{angle:+03d}_br{brightness:.2f}_co{contrast:.2f}"
                        
                        # Create augmented version
                        augmented_data = self.create_augmented_version(
                            image_rgb, all_detections, character_detections, 
                            serial_numbers, angle, brightness, contrast
                        )
                        
                        # Save annotations
                        output_paths = self.save_augmented_annotations(
                            image_path, original_size, augmented_data, aug_suffix
                        )
                        
                        # Update statistics
                        for char_data in augmented_data['character_detections']:
                            self.stats['character_count'][char_data['character']] += 1
                        
                        result = {
                            'original_image': str(image_path),
                            'augmented_image': output_paths['image_path'],
                            'augmentation_params': augmented_data['augmentation_params'],
                            'serial_numbers': [sn['serial_number'] for sn in augmented_data['serial_numbers']],
                            'character_count': len(augmented_data['character_detections']),
                            'output_files': output_paths
                        }
                        
                        results.append(result)
                        aug_count += 1
            
            processing_time = time.time() - start_time
            self.stats['total_images'] += 1
            self.stats['total_augmented'] += aug_count
            self.stats['images_with_serials'] += 1
            self.stats['total_serials_found'] += len(serial_numbers)
            self.stats['processing_times'].append(processing_time)
            
            print(f"   ✅ Created {aug_count} augmented versions")
            
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.stats['errors'].append(error_msg)
        
        return results
    
    def generate_character_frequency_report(self) -> Path:
        """
        Generate character frequency report for training
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"character_frequency_{timestamp}.csv"
        
        with open(csv_path, 'w') as f:
            f.write("Character,Count,Class_ID\n")
            for char, class_id in self.char_to_id.items():
                count = self.stats['character_count'][char]
                f.write(f"{char},{count},{class_id}\n")
        
        return csv_path
    
    def process_batch_with_augmentation(self, input_dir: str, file_pattern: str = "*") -> Dict:
        """
        Process batch of images with augmentation
        """
        print("🚀 Starting Batch Processing with Geometric Augmentation")
        print("=" * 70)
        
        # Get image files
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        if file_pattern == "*":
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            for ext in extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
        else:
            image_files = list(input_path.glob(file_pattern))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            return {}
        
        print(f"📁 Found {len(image_files)} images to process")
        print(f"📁 Will generate {len(self.rotation_angles) * len(self.brightness_factors) * len(self.contrast_factors) - 1} augmented versions per image")
        print(f"📁 Total expected outputs: {len(image_files) * (len(self.rotation_angles) * len(self.brightness_factors) * len(self.contrast_factors) - 1)}")
        print()
        
        # Process images
        all_results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end="")
            results = self.process_image_with_augmentation(image_path)
            all_results.extend(results)
        
        total_time = time.time() - start_time
        
        # Generate reports
        print("\n📊 Generating reports...")
        char_freq_path = self.generate_character_frequency_report()
        
        # Generate summary
        summary = {
            'processing_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_original_images': self.stats['total_images'],
                'total_augmented_images': self.stats['total_augmented'],
                'images_with_serials': self.stats['images_with_serials'],
                'total_serials_found': self.stats['total_serials_found'],
                'total_characters_annotated': sum(self.stats['character_count'].values())
            },
            'character_distribution': self.stats['character_count'],
            'augmentation_settings': {
                'rotation_angles': self.rotation_angles,
                'brightness_factors': self.brightness_factors,
                'contrast_factors': self.contrast_factors
            },
            'performance': {
                'total_processing_time': total_time,
                'average_per_image': total_time / len(image_files) if image_files else 0,
                'images_per_second': len(image_files) / total_time if total_time > 0 else 0
            },
            'output_paths': {
                'augmented_images': str(self.augmented_images_dir),
                'json_annotations': str(self.json_dir),
                'yolo_annotations': str(self.yolo_dir),
                'visual_annotations': str(self.visual_dir),
                'character_frequency_report': str(char_freq_path)
            }
        }
        
        # Save summary
        summary_path = self.reports_dir / f"augmentation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print("\n" + "=" * 70)
        print("🎯 AUGMENTED BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"📊 Original Images: {summary['processing_summary']['total_original_images']}")
        print(f"🔄 Augmented Images: {summary['processing_summary']['total_augmented']}")
        print(f"📝 Total Characters Annotated: {summary['processing_summary']['total_characters_annotated']}")
        print(f"🎯 Serial Numbers Found: {summary['processing_summary']['total_serials_found']}")
        print()
        print("📋 Character Distribution:")
        for char, count in sorted(self.stats['character_count'].items()):
            if count > 0:
                print(f"   {char}: {count}")
        print()
        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        print(f"⏱️  Average per Image: {summary['performance']['average_per_image']:.2f} seconds")
        print()
        print("📁 Output Files:")
        print(f"   🖼️  Augmented Images: {self.augmented_images_dir}")
        print(f"   📝 JSON Annotations: {self.json_dir}")
        print(f"   📝 YOLO Annotations: {self.yolo_dir}")
        print(f"   🎨 Visual Annotations: {self.visual_dir}")
        print(f"   📊 Character Frequency: {char_freq_path}")
        print(f"   📋 Summary Report: {summary_path}")
        print("=" * 70)
        
        return summary

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Geometric Augmented Serial Number Annotator')
    parser.add_argument('input_dir', help='Directory containing images to process')
    parser.add_argument('--output_dir', default='augmented_annotations_output', 
                       help='Output directory for annotations')
    parser.add_argument('--pattern', default='*', 
                       help='File pattern to match (e.g., "*.jpg", "fuel_*.png")')
    
    args = parser.parse_args()
    
    # Initialize annotator
    annotator = GeometricAugmentedAnnotator(output_dir=args.output_dir)
    
    # Process batch
    summary = annotator.process_batch_with_augmentation(args.input_dir, args.pattern)
    
    return summary

if __name__ == "__main__":
    # If no command line args, use default settings
    import sys
    if len(sys.argv) == 1:
        print("🔧 Running with default settings...")
        print("📁 Input directory: test_images")
        print("📁 Output directory: augmented_annotations_output")
        print()
        
        # Check if test_images exists
        if not Path("test_images").exists():
            print("❌ Error: test_images directory not found!")
            print("Usage: python geometric_augmented_annotator.py <input_directory>")
            sys.exit(1)
        
        # Run with defaults
        annotator = GeometricAugmentedAnnotator()
        summary = annotator.process_batch_with_augmentation("test_images")
    else:
        main()
