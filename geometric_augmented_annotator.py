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

class GeometricAugmentedAnnotator:
    """
    Advanced Serial Number Annotator with Geometric Augmentations
    Rotates images and adjusts bounding boxes accordingly for robust training data
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
        
        # Augmentation settings
        self.rotation_angles = [-15, -10, -5, 0, 5, 10, 15]  # Degrees
        self.scale_factors = [0.9, 1.0, 1.1]
        self.brightness_factors = [0.8, 1.0, 1.2]
        
        print("✅ Geometric augmented annotator initialized successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
    
    def rotate_point(self, point, angle, center):
        """Rotate a point around a center by given angle (in degrees)"""
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # Translate point to origin
        translated_x = point[0] - center[0]
        translated_y = point[1] - center[1]
        
        # Rotate
        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle
        
        # Translate back
        final_x = rotated_x + center[0]
        final_y = rotated_y + center[1]
        
        return [final_x, final_y]
    
    def rotate_bbox(self, bbox, angle, image_center, image_shape):
        """Rotate bounding box and return new coordinates"""
        x_min, y_min, x_max, y_max = bbox
        
        # Get all four corners of the bounding box
        corners = [
            [x_min, y_min],  # Top-left
            [x_max, y_min],  # Top-right
            [x_max, y_max],  # Bottom-right
            [x_min, y_max]   # Bottom-left
        ]
        
        # Rotate all corners
        rotated_corners = []
        for corner in corners:
            rotated_corner = self.rotate_point(corner, angle, image_center)
            rotated_corners.append(rotated_corner)
        
        # Find new bounding box from rotated corners
        x_coords = [corner[0] for corner in rotated_corners]
        y_coords = [corner[1] for corner in rotated_corners]
        
        new_x_min = max(0, min(x_coords))
        new_y_min = max(0, min(y_coords))
        new_x_max = min(image_shape[1], max(x_coords))
        new_y_max = min(image_shape[0], max(y_coords))
        
        return [int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)]
    
    def rotate_image_and_annotations(self, image, detections, angle):
        """Rotate image and adjust all bounding boxes accordingly"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to fit rotated image
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust rotation matrix for new image center
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Rotate image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        
        # Adjust bounding boxes
        rotated_detections = []
        new_center = (new_w // 2, new_h // 2)
        
        for detection in detections:
            # Transform bounding box coordinates
            bbox = detection['bbox']
            
            # Convert to points and transform
            points = np.array([
                [bbox[0], bbox[1], 1],  # Top-left
                [bbox[2], bbox[1], 1],  # Top-right  
                [bbox[2], bbox[3], 1],  # Bottom-right
                [bbox[0], bbox[3], 1]   # Bottom-left
            ]).T
            
            transformed_points = rotation_matrix @ points
            
            # Get new bounding box
            x_coords = transformed_points[0, :]
            y_coords = transformed_points[1, :]
            
            new_x_min = max(0, int(min(x_coords)))
            new_y_min = max(0, int(min(y_coords)))
            new_x_max = min(new_w, int(max(x_coords)))
            new_y_max = min(new_h, int(max(y_coords)))
            
            # Only keep detections that are still visible
            if new_x_max > new_x_min and new_y_max > new_y_min:
                rotated_detection = detection.copy()
                rotated_detection['bbox'] = [new_x_min, new_y_min, new_x_max, new_y_max]
                rotated_detections.append(rotated_detection)
        
        return rotated_image, rotated_detections, (new_w, new_h)
    
    def apply_brightness_augmentation(self, image, factor):
        """Apply brightness augmentation"""
        # Convert to float and apply brightness
        image_float = image.astype(np.float32) * factor
        # Clip to valid range and convert back
        return np.clip(image_float, 0, 255).astype(np.uint8)
    
    def apply_scale_augmentation(self, image, detections, scale_factor):
        """Apply scaling augmentation"""
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        
        # Resize image
        scaled_image = cv2.resize(image, (new_w, new_h))
        
        # Scale bounding boxes
        scaled_detections = []
        for detection in detections:
            bbox = detection['bbox']
            scaled_bbox = [
                int(bbox[0] * scale_factor),
                int(bbox[1] * scale_factor),
                int(bbox[2] * scale_factor),
                int(bbox[3] * scale_factor)
            ]
            
            scaled_detection = detection.copy()
            scaled_detection['bbox'] = scaled_bbox
            scaled_detections.append(scaled_detection)
        
        return scaled_image, scaled_detections, (new_w, new_h)
    
    def detect_text(self, image):
        """Detect all text regions using EasyOCR"""
        results = self.reader.readtext(image, detail=1, paragraph=False)
        
        detections = []
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
                
                detections.append({
                    'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                    'bbox_poly': bbox_poly,
                    'text': text.strip(),
                    'confidence': float(confidence)
                })
        
        return detections
    
    def filter_serial_numbers(self, detections):
        """Filter detections to find fuel bundle serial numbers"""
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
    
    def segment_characters(self, image, serial_detection):
        """Segment individual characters from serial number region"""
        bbox = serial_detection['bbox']
        x_min, y_min, x_max, y_max = bbox
        
        # Crop serial number region
        serial_crop = image[y_min:y_max, x_min:x_max]
        
        if serial_crop.size == 0:
            return []
        
        # Convert to grayscale for better character segmentation
        gray = cv2.cvtColor(serial_crop, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours by x-coordinate
        char_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small noise
            if w > 5 and h > 10:
                char_contours.append((x, y, w, h, contour))
        
        # Sort by x-coordinate (left to right)
        char_contours.sort(key=lambda x: x[0])
        
        # Create character annotations
        character_annotations = []
        serial_number = serial_detection.get('serial_number', '')
        
        for i, (rel_x, rel_y, w, h, contour) in enumerate(char_contours):
            # Convert relative coordinates to absolute
            abs_x_min = x_min + rel_x
            abs_y_min = y_min + rel_y
            abs_x_max = abs_x_min + w
            abs_y_max = abs_y_min + h
            
            # Get character if we know the serial number
            character = serial_number[i] if i < len(serial_number) else '?'
            
            character_annotations.append({
                'bbox': [abs_x_min, abs_y_min, abs_x_max, abs_y_max],
                'character': character,
                'position_in_serial': i,
                'contour': contour.tolist()
            })
        
        return character_annotations
    
    def process_image_with_augmentations(self, image_path):
        """Process image with multiple augmentations"""
        print(f"📸 Processing with augmentations: {image_path.name}")
        
        # Load original image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Detect text in original image
        all_detections = self.detect_text(image_rgb)
        serial_numbers = self.filter_serial_numbers(all_detections)
        
        if not serial_numbers:
            print(f"   ⚠️  No serial numbers found in {image_path.name}")
            return []
        
        augmentation_results = []
        aug_id = 0
        
        # Apply different augmentations
        for rotation_angle in self.rotation_angles:
            for scale_factor in self.scale_factors:
                for brightness_factor in self.brightness_factors:
                    aug_id += 1
                    
                    # Start with original image
                    current_image = image_rgb.copy()
                    current_detections = serial_numbers.copy()
                    current_size = (image_rgb.shape[1], image_rgb.shape[0])  # (width, height)
                    
                    # Apply brightness augmentation
                    if brightness_factor != 1.0:
                        current_image = self.apply_brightness_augmentation(current_image, brightness_factor)
                    
                    # Apply scaling augmentation
                    if scale_factor != 1.0:
                        current_image, current_detections, current_size = self.apply_scale_augmentation(
                            current_image, current_detections, scale_factor)
                    
                    # Apply rotation augmentation
                    if rotation_angle != 0:
                        current_image, current_detections, current_size = self.rotate_image_and_annotations(
                            current_image, current_detections, rotation_angle)
                    
                    # Skip if no detections remain after augmentation
                    if not current_detections:
                        continue
                    
                    # Generate character-level annotations
                    all_character_annotations = []
                    for serial_detection in current_detections:
                        char_annotations = self.segment_characters(current_image, serial_detection)
                        all_character_annotations.extend(char_annotations)
                    
                    # Create augmented filename
                    aug_suffix = f"_rot{rotation_angle}_scale{scale_factor:.1f}_bright{brightness_factor:.1f}"
                    aug_filename = f"{image_path.stem}{aug_suffix}.jpg"
                    
                    # Save augmented image
                    aug_image_path = self.augmented_images_dir / aug_filename
                    cv2.imwrite(str(aug_image_path), cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
                    
                    # Create annotation data
                    annotation_data = {
                        'image_info': {
                            'filename': aug_filename,
                            'original_filename': image_path.name,
                            'width': current_size[0],
                            'height': current_size[1],
                            'channels': 3
                        },
                        'augmentation_info': {
                            'rotation_angle': rotation_angle,
                            'scale_factor': scale_factor,
                            'brightness_factor': brightness_factor,
                            'augmentation_id': aug_id
                        },
                        'detection_summary': {
                            'total_text_detections': len(all_detections),
                            'serial_numbers_found': len(current_detections),
                            'characters_found': len(all_character_annotations)
                        },
                        'serial_numbers': current_detections,
                        'character_annotations': all_character_annotations,
                        'metadata': {
                            'created_at': datetime.now().isoformat(),
                            'detector_version': '2.0_augmented',
                            'patterns_supported': self.pattern_names
                        }
                    }
                    
                    # Save JSON annotation
                    json_path = self.json_dir / f"{image_path.stem}{aug_suffix}.json"
                    with open(json_path, 'w') as f:
                        json.dump(annotation_data, f, indent=2)
                    
                    # Save YOLO annotations (both word and character level)
                    self.save_yolo_annotations(current_size, current_detections, all_character_annotations, 
                                             image_path.stem + aug_suffix)
                    
                    # Save visual annotation
                    self.save_visual_annotation(current_image, current_detections, all_character_annotations,
                                              image_path.stem + aug_suffix)
                    
                    augmentation_results.append({
                        'augmentation_id': aug_id,
                        'filename': aug_filename,
                        'rotation': rotation_angle,
                        'scale': scale_factor,
                        'brightness': brightness_factor,
                        'serials_found': len(current_detections),
                        'characters_found': len(all_character_annotations)
                    })
                    
                    print(f"   ✅ Aug {aug_id}: rot={rotation_angle}°, scale={scale_factor}, bright={brightness_factor} "
                          f"→ {len(current_detections)} serials, {len(all_character_annotations)} chars")
        
        return augmentation_results
    
    def save_yolo_annotations(self, image_size, serial_detections, character_annotations, base_filename):
        """Save YOLO format annotations for both words and characters"""
        
        # Word-level YOLO (class 0 = serial number)
        word_yolo_path = self.yolo_dir / f"{base_filename}_words.txt"
        with open(word_yolo_path, 'w') as f:
            for detection in serial_detections:
                x_min, y_min, x_max, y_max = detection['bbox']
                center_x = (x_min + x_max) / 2 / image_size[0]
                center_y = (y_min + y_max) / 2 / image_size[1]
                width = (x_max - x_min) / image_size[0]
                height = (y_max - y_min) / image_size[1]
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Character-level YOLO (class based on character type)
        char_yolo_path = self.yolo_dir / f"{base_filename}_characters.txt"
        char_to_id = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
            'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 
            'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22,
            'W': 23, 'X': 24, 'Y': 25, 'Z': 26, '0': 27, '1': 28, '2': 29,
            '3': 30, '4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36
        }
        
        with open(char_yolo_path, 'w') as f:
            for char_annotation in character_annotations:
                char = char_annotation['character']
                if char in char_to_id:
                    x_min, y_min, x_max, y_max = char_annotation['bbox']
                    center_x = (x_min + x_max) / 2 / image_size[0]
                    center_y = (y_min + y_max) / 2 / image_size[1]
                    width = (x_max - x_min) / image_size[0]
                    height = (y_max - y_min) / image_size[1]
                    class_id = char_to_id[char]
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    def save_visual_annotation(self, image, serial_detections, character_annotations, base_filename):
        """Save visual annotation with both word and character bounding boxes"""
        result_image = image.copy()
        
        # Draw serial number bounding boxes in red
        for detection in serial_detections:
            x_min, y_min, x_max, y_max = detection['bbox']
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            
            # Add serial number label
            label = f"SERIAL: {detection.get('serial_number', 'Unknown')}"
            cv2.putText(result_image, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw character bounding boxes in green
        for char_annotation in character_annotations:
            x_min, y_min, x_max, y_max = char_annotation['bbox']
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add character label
            char = char_annotation['character']
            cv2.putText(result_image, char, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save visual annotation
        visual_path = self.visual_dir / f"{base_filename}_annotated.jpg"
        cv2.imwrite(str(visual_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    def process_batch_with_augmentations(self, input_dir):
        """Process batch of images with augmentations"""
        print("🚀 Starting Batch Processing with Geometric Augmentations")
        print("=" * 70)
        
        # Get image files
        input_path = Path(input_dir)
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            return
        
        print(f"📁 Found {len(image_files)} images to process")
        print(f"🔄 Will generate {len(self.rotation_angles) * len(self.scale_factors) * len(self.brightness_factors)} augmentations per image")
        print()
        
        all_results = []
        total_augmentations = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end="")
            try:
                results = self.process_image_with_augmentations(image_path)
                all_results.extend(results)
                total_augmentations += len(results)
            except Exception as e:
                print(f"   ❌ Error processing {image_path.name}: {e}")
        
        # Generate summary report
        self.generate_augmentation_report(all_results, total_augmentations)
        
        print("\n" + "=" * 70)
        print("🎯 AUGMENTED BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"📊 Total Original Images: {len(image_files)}")
        print(f"📊 Total Augmented Images: {total_augmentations}")
        print(f"📊 Augmentation Factor: {total_augmentations / len(image_files) if image_files else 0:.1f}x")
        print(f"📁 Output Directory: {self.output_dir}")
        print("=" * 70)
    
    def generate_augmentation_report(self, results, total_augmentations):
        """Generate detailed augmentation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV report
        csv_path = self.reports_dir / f"augmentation_report_{timestamp}.csv"
        with open(csv_path, 'w') as f:
            f.write("Aug_ID,Filename,Rotation,Scale,Brightness,Serials_Found,Characters_Found\n")
            for result in results:
                f.write(f"{result['augmentation_id']},{result['filename']},{result['rotation']},"
                       f"{result['scale']},{result['brightness']},{result['serials_found']},{result['characters_found']}\n")
        
        # JSON summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_augmentations': total_augmentations,
            'augmentation_settings': {
                'rotation_angles': self.rotation_angles,
                'scale_factors': self.scale_factors,
                'brightness_factors': self.brightness_factors
            },
            'results': results
        }
        
        json_path = self.reports_dir / f"augmentation_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "test_images"
    
    print("🔧 Geometric Augmented Serial Number Annotator")
    print("📋 Features:")
    print("   - Rotation with bbox adjustment")
    print("   - Scale augmentation") 
    print("   - Brightness variation")
    print("   - Character-level segmentation")
    print("   - Word + Character YOLO annotations")
    print()
    
    # Initialize annotator
    annotator = GeometricAugmentedAnnotator()
    
    # Process with augmentations
    annotator.process_batch_with_augmentations(input_dir)

if __name__ == "__main__":
    main()
