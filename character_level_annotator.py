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

class CharacterLevelAnnotator:
    """
    Character-Level Serial Number Annotator for Fuel Bundle Images
    Generates annotations for:
    1. Individual characters (letters/digits)
    2. Complete serial numbers
    3. Pattern classification
    
    Output formats:
    - JSON with character-level and word-level annotations
    - YOLO format for both characters and words
    - Visual annotations showing individual character boxes
    """
    
    def __init__(self, output_dir="character_annotations_output"):
        print("🚀 Initializing Character-Level Serial Number Annotator...")
        
        # Create output directory structure
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.json_dir = self.output_dir / "json_annotations"
        self.yolo_chars_dir = self.output_dir / "yolo_characters"
        self.yolo_words_dir = self.output_dir / "yolo_words"
        self.visual_dir = self.output_dir / "visual_annotations"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.json_dir, self.yolo_chars_dir, self.yolo_words_dir, self.visual_dir, self.reports_dir]:
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
            r'^[A-Z]{3}\d{3}[A-Z]$'   # AAAnnnA (JFH772B)
        ]
        
        self.pattern_names = ['AAnnnnA', 'AAnnnAn', 'AAAnnnA']
        
        # Character class mapping for YOLO
        self.char_classes = {
            # Letters
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
            'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
            # Numbers
            '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35
        }
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'images_with_serials': 0,
            'total_serials_found': 0,
            'total_characters_found': 0,
            'pattern_counts': {name: 0 for name in self.pattern_names},
            'character_counts': {char: 0 for char in self.char_classes.keys()},
            'processing_times': [],
            'errors': []
        }
        
        print("✅ Character-level annotator initialized successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"🔤 Character classes: {len(self.char_classes)} (A-Z: 0-25, 0-9: 26-35)")
    
    def get_image_files(self, input_dir, extensions=None):
        """Get all image files from input directory"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def preprocess_image(self, image_path):
        """Enhanced preprocessing for fuel bundle images"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large
        h, w = image_rgb.shape[:2]
        original_size = (w, h)
        
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        # Enhance contrast
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb, original_size
    
    def detect_text_and_characters(self, image):
        """Detect text regions and individual characters using EasyOCR"""
        # Word-level detection
        word_results = self.reader.readtext(image, detail=1, paragraph=False)
        
        # Character-level detection (width_ths=0.1 for better character separation)
        char_results = self.reader.readtext(image, detail=1, paragraph=False, width_ths=0.1)
        
        word_detections = []
        char_detections = []
        
        # Process word-level results
        for bbox, text, confidence in word_results:
            if confidence > 0.1:
                bbox_np = np.array(bbox, dtype=np.int32)
                x_min, y_min = bbox_np[:, 0].min(), bbox_np[:, 1].min()
                x_max, y_max = bbox_np[:, 0].max(), bbox_np[:, 1].max()
                
                if isinstance(bbox, np.ndarray):
                    bbox_poly = bbox.tolist()
                else:
                    bbox_poly = bbox
                
                word_detections.append({
                    'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                    'bbox_poly': bbox_poly,
                    'text': text.strip(),
                    'confidence': float(confidence),
                    'type': 'word'
                })
        
        # Process character-level results
        for bbox, text, confidence in char_results:
            if confidence > 0.3 and len(text.strip()) == 1:  # Single character with higher confidence
                bbox_np = np.array(bbox, dtype=np.int32)
                x_min, y_min = bbox_np[:, 0].min(), bbox_np[:, 1].min()
                x_max, y_max = bbox_np[:, 0].max(), bbox_np[:, 1].max()
                
                if isinstance(bbox, np.ndarray):
                    bbox_poly = bbox.tolist()
                else:
                    bbox_poly = bbox
                
                char = text.strip().upper()
                if char in self.char_classes:
                    char_detections.append({
                        'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                        'bbox_poly': bbox_poly,
                        'text': char,
                        'confidence': float(confidence),
                        'type': 'character',
                        'class_id': self.char_classes[char]
                    })
        
        return word_detections, char_detections
    
    def segment_characters_from_word(self, word_detection, image):
        """Segment individual characters from a detected word"""
        word_bbox = word_detection['bbox']
        x_min, y_min, x_max, y_max = word_bbox
        word_text = word_detection['text'].upper()
        clean_text = re.sub(r'[^A-Z0-9]', '', word_text)
        
        # Extract word region
        word_region = image[y_min:y_max, x_min:x_max]
        
        # Convert to grayscale for better character segmentation
        gray = cv2.cvtColor(word_region, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours for character segmentation
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        character_annotations = []
        
        # If we have the expected number of contours, create character boxes
        if len(contours) == len(clean_text):
            for i, contour in enumerate(contours):
                char_x, char_y, char_w, char_h = cv2.boundingRect(contour)
                
                # Convert relative coordinates to absolute
                abs_x_min = x_min + char_x
                abs_y_min = y_min + char_y
                abs_x_max = abs_x_min + char_w
                abs_y_max = abs_y_min + char_h
                
                if i < len(clean_text):
                    char = clean_text[i]
                    if char in self.char_classes:
                        character_annotations.append({
                            'bbox': [abs_x_min, abs_y_min, abs_x_max, abs_y_max],
                            'text': char,
                            'confidence': word_detection['confidence'] * 0.9,  # Slightly lower than word confidence
                            'type': 'character_segmented',
                            'class_id': self.char_classes[char],
                            'parent_word': word_detection['text'],
                            'position_in_word': i
                        })
        
        # Fallback: divide word bbox equally if contour method fails
        if not character_annotations and len(clean_text) > 0:
            char_width = (x_max - x_min) / len(clean_text)
            
            for i, char in enumerate(clean_text):
                if char in self.char_classes:
                    char_x_min = int(x_min + i * char_width)
                    char_x_max = int(x_min + (i + 1) * char_width)
                    
                    character_annotations.append({
                        'bbox': [char_x_min, y_min, char_x_max, y_max],
                        'text': char,
                        'confidence': word_detection['confidence'] * 0.8,  # Lower confidence for estimated boxes
                        'type': 'character_estimated',
                        'class_id': self.char_classes[char],
                        'parent_word': word_detection['text'],
                        'position_in_word': i
                    })
        
        return character_annotations
    
    def filter_serial_numbers(self, word_detections):
        """Filter detections to find fuel bundle serial numbers"""
        serial_numbers = []
        
        for detection in word_detections:
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
                            break
            
            # OCR error correction
            if len(clean_text) >= 6 and len(clean_text) <= 8:
                ocr_fixes = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G', '2': 'Z'}
                
                for wrong, right in ocr_fixes.items():
                    fixed_text = clean_text.replace(wrong, right)
                    if len(fixed_text) == 7:
                        for i, pattern in enumerate(self.serial_patterns):
                            if re.match(pattern, fixed_text):
                                detection_copy = detection.copy()
                                detection_copy.update({
                                    'serial_number': fixed_text,
                                    'pattern': self.pattern_names[i],
                                    'pattern_index': i,
                                    'original_text': text,
                                    'ocr_corrected': True
                                })
                                if detection_copy not in serial_numbers:
                                    serial_numbers.append(detection_copy)
                                break
        
        return serial_numbers
    
    def save_json_annotation(self, image_path, image_size, word_detections, char_detections, serial_numbers, character_annotations):
        """Save comprehensive annotations in JSON format"""
        annotation = {
            'image_info': {
                'filename': image_path.name,
                'width': image_size[0],
                'height': image_size[1],
                'channels': 3
            },
            'detection_summary': {
                'total_word_detections': len(word_detections),
                'total_character_detections': len(char_detections),
                'total_character_annotations': len(character_annotations),
                'serial_numbers_found': len(serial_numbers),
                'patterns_detected': list(set([sn['pattern'] for sn in serial_numbers]))
            },
            'word_level_detections': word_detections,
            'character_level_detections': char_detections,
            'character_annotations': character_annotations,
            'serial_numbers': serial_numbers,
            'character_classes': self.char_classes,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'detector_version': '2.0_character_level',
                'patterns_supported': self.pattern_names,
                'annotation_types': ['word_level', 'character_level', 'character_segmented', 'character_estimated']
            }
        }
        
        json_path = self.json_dir / f"{image_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        return json_path
    
    def save_yolo_annotations(self, image_path, image_size, serial_numbers, character_annotations):
        """Save YOLO format annotations for both words and characters"""
        
        # Save word-level YOLO annotations
        yolo_words_path = self.yolo_words_dir / f"{image_path.stem}.txt"
        with open(yolo_words_path, 'w') as f:
            for sn in serial_numbers:
                x_min, y_min, x_max, y_max = sn['bbox']
                center_x = (x_min + x_max) / 2 / image_size[0]
                center_y = (y_min + y_max) / 2 / image_size[1]
                width = (x_max - x_min) / image_size[0]
                height = (y_max - y_min) / image_size[1]
                
                # Class ID 0 for serial number
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Save character-level YOLO annotations
        yolo_chars_path = self.yolo_chars_dir / f"{image_path.stem}.txt"
        with open(yolo_chars_path, 'w') as f:
            for char_ann in character_annotations:
                x_min, y_min, x_max, y_max = char_ann['bbox']
                center_x = (x_min + x_max) / 2 / image_size[0]
                center_y = (y_min + y_max) / 2 / image_size[1]
                width = (x_max - x_min) / image_size[0]
                height = (y_max - y_min) / image_size[1]
                
                class_id = char_ann['class_id']
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        return yolo_words_path, yolo_chars_path
    
    def save_visual_annotation(self, image_path, image, word_detections, char_detections, serial_numbers, character_annotations):
        """Save visual annotation with bounding boxes for words and characters"""
        result_image = image.copy()
        
        # Draw all word detections in blue
        for detection in word_detections:
            x_min, y_min, x_max, y_max = detection['bbox']
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (100, 149, 237), 2)
            
            text_label = f"{detection['text']} ({detection['confidence']:.2f})"
            cv2.putText(result_image, text_label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 149, 237), 1)
        
        # Draw serial numbers in red (thick)
        for sn in serial_numbers:
            x_min, y_min, x_max, y_max = sn['bbox']
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)
            
            label = f"SERIAL: {sn['serial_number']}"
            label_pos = (x_min, max(y_min - 40, 30))
            
            # Background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result_image,
                         (label_pos[0] - 5, label_pos[1] - text_height - 5),
                         (label_pos[0] + text_width + 5, label_pos[1] + baseline + 5),
                         (255, 0, 0), -1)
            
            cv2.putText(result_image, label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            pattern_label = f"Pattern: {sn['pattern']}"
            pattern_pos = (x_min, y_max + 30)
            cv2.putText(result_image, pattern_label, pattern_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw character-level annotations in green
        for char_ann in character_annotations:
            x_min, y_min, x_max, y_max = char_ann['bbox']
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add character label
            char_label = f"{char_ann['text']}"
            cv2.putText(result_image, char_label, (x_min + 2, y_min - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add legend
        legend_y = 30
        cv2.putText(result_image, "Blue: All Text | Red: Serial Numbers | Green: Characters", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save image with compression
        visual_path = self.visual_dir / f"{image_path.stem}_character_annotated.jpg"
        cv2.imwrite(str(visual_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), 
                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return visual_path
    
    def process_single_image(self, image_path):
        """Process a single image and generate character-level annotations"""
        print(f"📸 Processing: {image_path.name}")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            image, original_size = self.preprocess_image(image_path)
            
            # Detect text at word and character level
            word_detections, char_detections = self.detect_text_and_characters(image)
            
            # Filter serial numbers
            serial_numbers = self.filter_serial_numbers(word_detections)
            
            # Generate character-level annotations for serial numbers
            character_annotations = []
            for serial in serial_numbers:
                char_anns = self.segment_characters_from_word(serial, image)
                character_annotations.extend(char_anns)
            
            # Save annotations
            json_path = self.save_json_annotation(image_path, original_size, word_detections, 
                                                char_detections, serial_numbers, character_annotations)
            yolo_words_path, yolo_chars_path = self.save_yolo_annotations(image_path, original_size, 
                                                                        serial_numbers, character_annotations)
            visual_path = self.save_visual_annotation(image_path, image, word_detections, 
                                                    char_detections, serial_numbers, character_annotations)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_images'] += 1
            if serial_numbers:
                self.stats['images_with_serials'] += 1
                self.stats['total_serials_found'] += len(serial_numbers)
                for sn in serial_numbers:
                    self.stats['pattern_counts'][sn['pattern']] += 1
            
            self.stats['total_characters_found'] += len(character_annotations)
            for char_ann in character_annotations:
                char = char_ann['text']
                if char in self.stats['character_counts']:
                    self.stats['character_counts'][char] += 1
            
            self.stats['processing_times'].append(processing_time)
            
            result = {
                'image_path': str(image_path),
                'success': True,
                'serial_numbers_found': len(serial_numbers),
                'characters_found': len(character_annotations),
                'serial_numbers': [sn['serial_number'] for sn in serial_numbers],
                'processing_time': processing_time,
                'output_files': {
                    'json': str(json_path),
                    'yolo_words': str(yolo_words_path),
                    'yolo_chars': str(yolo_chars_path),
                    'visual': str(visual_path)
                }
            }
            
            print(f"   ✅ Found {len(serial_numbers)} serial(s) with {len(character_annotations)} characters: {[sn['serial_number'] for sn in serial_numbers]}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            
            return {
                'image_path': str(image_path),
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time
            }
    
    def generate_report(self, results):
        """Generate detailed processing report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate CSV report
        csv_path = self.reports_dir / f"character_annotation_report_{timestamp}.csv"
        with open(csv_path, 'w') as f:
            f.write("Image,Success,Serials_Found,Characters_Found,Serial_Numbers,Processing_Time,Error\n")
            for result in results:
                image_name = Path(result['image_path']).name
                success = result['success']
                serials_found = result.get('serial_numbers_found', 0)
                chars_found = result.get('characters_found', 0)
                serial_numbers = ';'.join(result.get('serial_numbers', []))
                processing_time = result['processing_time']
                error = result.get('error', '')
                
                f.write(f"{image_name},{success},{serials_found},{chars_found},\"{serial_numbers}\",{processing_time:.3f},\"{error}\"\n")
        
        # Generate character frequency report
        char_freq_path = self.reports_dir / f"character_frequency_{timestamp}.csv"
        with open(char_freq_path, 'w') as f:
            f.write("Character,Count,Class_ID\n")
            for char, count in sorted(self.stats['character_counts'].items()):
                class_id = self.char_classes[char]
                f.write(f"{char},{count},{class_id}\n")
        
        # Generate JSON summary report
        avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        summary_report = {
            'processing_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_images': self.stats['total_images'],
                'successful_images': sum(1 for r in results if r['success']),
                'failed_images': sum(1 for r in results if not r['success']),
                'images_with_serials': self.stats['images_with_serials'],
                'total_serials_found': self.stats['total_serials_found'],
                'total_characters_found': self.stats['total_characters_found'],
                'success_rate': (sum(1 for r in results if r['success']) / len(results)) * 100 if results else 0,
                'detection_rate': (self.stats['images_with_serials'] / self.stats['total_images']) * 100 if self.stats['total_images'] else 0
            },
            'pattern_statistics': self.stats['pattern_counts'],
            'character_statistics': self.stats['character_counts'],
            'character_classes': self.char_classes,
            'performance': {
                'average_processing_time': avg_processing_time,
                'total_processing_time': sum(self.stats['processing_times']),
                'fastest_image': min(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'slowest_image': max(self.stats['processing_times']) if self.stats['processing_times'] else 0
            },
            'errors': self.stats['errors'],
            'detailed_results': results
        }
        
        json_report_path = self.reports_dir / f"character_annotation_summary_{timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        return csv_path, char_freq_path, json_report_path, summary_report
    
    def process_batch(self, input_dir, file_pattern="*"):
        """Process a batch of images"""
        print("🚀 Starting Character-Level Serial Number Annotation")
        print("=" * 70)
        
        # Get image files
        try:
            if file_pattern == "*":
                image_files = self.get_image_files(input_dir)
            else:
                input_path = Path(input_dir)
                image_files = list(input_path.glob(file_pattern))
        except Exception as e:
            print(f"❌ Error finding images: {e}")
            return
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            return
        
        print(f"📁 Found {len(image_files)} images to process")
        print(f"📁 Input directory: {input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        
        # Process images
        results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end="")
            result = self.process_single_image(image_path)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Generate reports
        print("\n📊 Generating reports...")
        csv_path, char_freq_path, json_path, summary = self.generate_report(results)
        
        # Print summary
        print("\n" + "=" * 70)
        print("🎯 CHARACTER-LEVEL ANNOTATION COMPLETE")
        print("=" * 70)
        print(f"📊 Total Images: {summary['processing_summary']['total_images']}")
        print(f"✅ Successful: {summary['processing_summary']['successful_images']}")
        print(f"❌ Failed: {summary['processing_summary']['failed_images']}")
        print(f"🔍 Images with Serials: {summary['processing_summary']['images_with_serials']}")
        print(f"📝 Total Serials Found: {summary['processing_summary']['total_serials_found']}")
        print(f"🔤 Total Characters Found: {summary['processing_summary']['total_characters_found']}")
        print(f"📈 Success Rate: {summary['processing_summary']['success_rate']:.1f}%")
        print(f"📈 Detection Rate: {summary['processing_summary']['detection_rate']:.1f}%")
        print()
        print("📋 Pattern Distribution:")
        for pattern, count in summary['pattern_statistics'].items():
            print(f"   {pattern}: {count}")
        print()
        print("🔤 Top Character Frequencies:")
        sorted_chars = sorted(summary['character_statistics'].items(), key=lambda x: x[1], reverse=True)[:10]
        for char, count in sorted_chars:
            if count > 0:
                print(f"   {char}: {count} (class {self.char_classes[char]})")
        print()
        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        print(f"⏱️  Average per Image: {summary['performance']['average_processing_time']:.2f} seconds")
        print()
        print("📁 Output Files:")
        print(f"   📊 CSV Report: {csv_path}")
        print(f"   📊 Character Frequency: {char_freq_path}")
        print(f"   📊 JSON Report: {json_path}")
        print(f"   📝 JSON Annotations: {self.json_dir}")
        print(f"   📝 YOLO Words: {self.yolo_words_dir}")
        print(f"   📝 YOLO Characters: {self.yolo_chars_dir}")
        print(f"   🖼️  Visual Annotations: {self.visual_dir}")
        print("=" * 70)
        
        return results, summary

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Character-Level Serial Number Annotator for Fuel Bundle Images')
    parser.add_argument('input_dir', help='Directory containing images to process')
    parser.add_argument('--output_dir', default='character_annotations_output', help='Output directory for annotations')
    parser.add_argument('--pattern', default='*', help='File pattern to match (e.g., "*.jpg", "fuel_*.png")')
    
    args = parser.parse_args()
    
    # Initialize annotator
    annotator = CharacterLevelAnnotator(output_dir=args.output_dir)
    
    # Process batch
    results, summary = annotator.process_batch(args.input_dir, args.pattern)
    
    return results, summary

if __name__ == "__main__":
    # If no command line args, use default settings
    import sys
    if len(sys.argv) == 1:
        print("🔧 Running with default settings...")
        print("📁 Input directory: test_images_2")
        print("📁 Output directory: character_annotations_output")
        print()
        
        # Check if test_images_2 exists
        if not Path("test_images_2").exists():
            print("❌ Error: test_images_2 directory not found!")
            print("Usage: python character_level_annotator.py <input_directory>")
            sys.exit(1)
        
        # Run with defaults
        annotator = CharacterLevelAnnotator()
        results, summary = annotator.process_batch("test_images_2")
    else:
        main()
