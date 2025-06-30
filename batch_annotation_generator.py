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

class BatchSerialAnnotator:
    """
    Batch Serial Number Annotator for Fuel Bundle Images
    Processes multiple images and generates annotations in various formats:
    - JSON annotations with bounding boxes
    - YOLO format annotations
    - Visual annotations with bounding boxes
    - CSV summary report
    """
    
    def __init__(self, output_dir="annotations_output"):
        print("🚀 Initializing Batch Serial Number Annotator...")
        
        # Create output directory structure
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.json_dir = self.output_dir / "json_annotations"
        self.yolo_dir = self.output_dir / "yolo_annotations"
        self.visual_dir = self.output_dir / "visual_annotations"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.json_dir, self.yolo_dir, self.visual_dir, self.reports_dir]:
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
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'images_with_serials': 0,
            'total_serials_found': 0,
            'pattern_counts': {name: 0 for name in self.pattern_names},
            'processing_times': [],
            'errors': []
        }
        
        print("✅ Batch annotator initialized successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
    
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
    
    def save_json_annotation(self, image_path, image_size, all_detections, serial_numbers):
        """Save annotations in JSON format"""
        annotation = {
            'image_info': {
                'filename': image_path.name,
                'width': image_size[0],
                'height': image_size[1],
                'channels': 3
            },
            'detection_summary': {
                'total_text_detections': len(all_detections),
                'serial_numbers_found': len(serial_numbers),
                'patterns_detected': list(set([sn['pattern'] for sn in serial_numbers]))
            },
            'all_text_detections': all_detections,
            'serial_numbers': serial_numbers,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'detector_version': '1.0',
                'patterns_supported': self.pattern_names
            }
        }
        
        json_path = self.json_dir / f"{image_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        return json_path
    
    def save_yolo_annotation(self, image_path, image_size, serial_numbers):
        """Save annotations in YOLO format"""
        yolo_path = self.yolo_dir / f"{image_path.stem}.txt"
        
        with open(yolo_path, 'w') as f:
            for sn in serial_numbers:
                # Convert bbox to YOLO format (normalized center_x, center_y, width, height)
                x_min, y_min, x_max, y_max = sn['bbox']
                
                center_x = (x_min + x_max) / 2 / image_size[0]
                center_y = (y_min + y_max) / 2 / image_size[1]
                width = (x_max - x_min) / image_size[0]
                height = (y_max - y_min) / image_size[1]
                
                # Class ID (0 for serial number)
                class_id = 0
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        return yolo_path
    
    def save_visual_annotation(self, image_path, image, all_detections, serial_numbers):
        """Save visual annotation with bounding boxes"""
        result_image = image.copy()
        
        # Draw all text detections in blue
        for detection in all_detections:
            x_min, y_min, x_max, y_max = detection['bbox']
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (100, 149, 237), 2)
            
            # Add text label
            text_label = f"{detection['text']} ({detection['confidence']:.2f})"
            cv2.putText(result_image, text_label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 149, 237), 1)
        
        # Draw serial numbers in red
        for sn in serial_numbers:
            x_min, y_min, x_max, y_max = sn['bbox']
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)
            
            # Add serial number label
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
            
            # Pattern info
            pattern_label = f"Pattern: {sn['pattern']}"
            pattern_pos = (x_min, y_max + 30)
            cv2.putText(result_image, pattern_label, pattern_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Save image
        visual_path = self.visual_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(visual_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        return visual_path
    
    def process_single_image(self, image_path):
        """Process a single image and generate all annotations"""
        print(f"📸 Processing: {image_path.name}")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            image, original_size = self.preprocess_image(image_path)
            
            # Detect text
            all_detections = self.detect_text(image)
            
            # Filter serial numbers
            serial_numbers = self.filter_serial_numbers(all_detections)
            
            # Save annotations
            json_path = self.save_json_annotation(image_path, original_size, all_detections, serial_numbers)
            yolo_path = self.save_yolo_annotation(image_path, original_size, serial_numbers)
            visual_path = self.save_visual_annotation(image_path, image, all_detections, serial_numbers)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_images'] += 1
            if serial_numbers:
                self.stats['images_with_serials'] += 1
                self.stats['total_serials_found'] += len(serial_numbers)
                for sn in serial_numbers:
                    self.stats['pattern_counts'][sn['pattern']] += 1
            
            self.stats['processing_times'].append(processing_time)
            
            result = {
                'image_path': str(image_path),
                'success': True,
                'serial_numbers_found': len(serial_numbers),
                'serial_numbers': [sn['serial_number'] for sn in serial_numbers],
                'processing_time': processing_time,
                'output_files': {
                    'json': str(json_path),
                    'yolo': str(yolo_path),
                    'visual': str(visual_path)
                }
            }
            
            print(f"   ✅ Found {len(serial_numbers)} serial(s): {[sn['serial_number'] for sn in serial_numbers]}")
            
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
        csv_path = self.reports_dir / f"batch_annotation_report_{timestamp}.csv"
        with open(csv_path, 'w') as f:
            f.write("Image,Success,Serials_Found,Serial_Numbers,Processing_Time,Error\n")
            for result in results:
                image_name = Path(result['image_path']).name
                success = result['success']
                serials_found = result.get('serial_numbers_found', 0)
                serial_numbers = ';'.join(result.get('serial_numbers', []))
                processing_time = result['processing_time']
                error = result.get('error', '')
                
                f.write(f"{image_name},{success},{serials_found},\"{serial_numbers}\",{processing_time:.3f},\"{error}\"\n")
        
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
                'success_rate': (sum(1 for r in results if r['success']) / len(results)) * 100 if results else 0,
                'detection_rate': (self.stats['images_with_serials'] / self.stats['total_images']) * 100 if self.stats['total_images'] else 0
            },
            'pattern_statistics': self.stats['pattern_counts'],
            'performance': {
                'average_processing_time': avg_processing_time,
                'total_processing_time': sum(self.stats['processing_times']),
                'fastest_image': min(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'slowest_image': max(self.stats['processing_times']) if self.stats['processing_times'] else 0
            },
            'errors': self.stats['errors'],
            'detailed_results': results
        }
        
        json_report_path = self.reports_dir / f"batch_annotation_summary_{timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        return csv_path, json_report_path, summary_report
    
    def process_batch(self, input_dir, file_pattern="*"):
        """Process a batch of images"""
        print("🚀 Starting Batch Serial Number Annotation")
        print("=" * 60)
        
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
        csv_path, json_path, summary = self.generate_report(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"📊 Total Images: {summary['processing_summary']['total_images']}")
        print(f"✅ Successful: {summary['processing_summary']['successful_images']}")
        print(f"❌ Failed: {summary['processing_summary']['failed_images']}")
        print(f"🔍 Images with Serials: {summary['processing_summary']['images_with_serials']}")
        print(f"📝 Total Serials Found: {summary['processing_summary']['total_serials_found']}")
        print(f"📈 Success Rate: {summary['processing_summary']['success_rate']:.1f}%")
        print(f"📈 Detection Rate: {summary['processing_summary']['detection_rate']:.1f}%")
        print()
        print("📋 Pattern Distribution:")
        for pattern, count in summary['pattern_statistics'].items():
            print(f"   {pattern}: {count}")
        print()
        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        print(f"⏱️  Average per Image: {summary['performance']['average_processing_time']:.2f} seconds")
        print()
        print("📁 Output Files:")
        print(f"   📊 CSV Report: {csv_path}")
        print(f"   📊 JSON Report: {json_path}")
        print(f"   📝 JSON Annotations: {self.json_dir}")
        print(f"   📝 YOLO Annotations: {self.yolo_dir}")
        print(f"   🖼️  Visual Annotations: {self.visual_dir}")
        print("=" * 60)
        
        return results, summary

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Batch Serial Number Annotator for Fuel Bundle Images')
    parser.add_argument('input_dir', help='Directory containing images to process')
    parser.add_argument('--output_dir', default='annotations_output', help='Output directory for annotations')
    parser.add_argument('--pattern', default='*', help='File pattern to match (e.g., "*.jpg", "fuel_*.png")')
    
    args = parser.parse_args()
    
    # Initialize annotator
    annotator = BatchSerialAnnotator(output_dir=args.output_dir)
    
    # Process batch
    results, summary = annotator.process_batch(args.input_dir, args.pattern)
    
    return results, summary

if __name__ == "__main__":
    # If no command line args, use default settings
    import sys
    if len(sys.argv) == 1:
        print("🔧 Running with default settings...")
        print("📁 Input directory: synthetic_dataset")
        print("📁 Output directory: annotations_output")
        print()
        
        # Check if synthetic_dataset exists
        if not Path("synthetic_dataset").exists():
            print("❌ Error: synthetic_dataset directory not found!")
            print("Usage: python batch_annotation_generator.py <input_directory>")
            sys.exit(1)
        
        # Run with defaults
        annotator = BatchSerialAnnotator()
        results, summary = annotator.process_batch("synthetic_dataset")
    else:
        main()
