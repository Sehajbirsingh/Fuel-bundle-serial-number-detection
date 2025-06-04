import torch
import cv2
import numpy as np
import easyocr
import re
from pathlib import Path
import time

class FuelBundleSerialDetector:
    """
    Fuel Bundle Serial Number Detector
    Detects 7-character serial numbers in formats:
    1. AAnnnnA (e.g., FD8764S)
    2. AAnnnAn (e.g., GFD887E)
    3. AAAnnnA
    """
    
    def __init__(self):
        print("Initializing Fuel Bundle Serial Number Detector...")
        print("Loading PyTorch-based OCR model...")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"🎯 CUDA detected! GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, using CPU")
        
        # Initialize EasyOCR (PyTorch-based) for English + numbers
        self.reader = easyocr.Reader(['en'], gpu=cuda_available)
        
        # Define exact serial number patterns (7 characters)
        self.serial_patterns = [
            r'^[A-Z]{2}\d{4}[A-Z]$',  # AAnnnnA (FD8764S)
            r'^[A-Z]{2}\d{3}[A-Z]\d$', # AAnnnAn (GFD887E) 
            r'^[A-Z]{3}\d{3}[A-Z]$'   # AAAnnnA
        ]
        
        print("✅ Model loaded successfully!")
        print(f"🔧 Using device: {'GPU' if cuda_available else 'CPU'}")
    
    def preprocess_image(self, image_path):
        """Enhanced preprocessing for fuel bundle images"""
        print(f"📸 Loading image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"❌ Could not load image: {image_path}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhance image for better OCR
        # 1. Resize if too large
        h, w = image_rgb.shape[:2]
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        # 2. Enhance contrast using CLAHE
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        print("✅ Image preprocessing completed")
        return enhanced_rgb
    
    def detect_text(self, image):
        """Detect all text regions using DBNet-based detection"""
        print("🔍 Detecting text regions...")
        
        # Run OCR detection and recognition
        results = self.reader.readtext(image, detail=1, paragraph=False)
        
        detections = []
        for bbox, text, confidence in results:
            if confidence > 0.1:  # Lower threshold to catch more text
                # Convert bbox to standardized format
                bbox_np = np.array(bbox, dtype=np.int32)
                x_min, y_min = bbox_np[:, 0].min(), bbox_np[:, 1].min()
                x_max, y_max = bbox_np[:, 0].max(), bbox_np[:, 1].max()
                
                detections.append({
                    'bbox': [[x_min, y_min], [x_max, y_max]],
                    'bbox_poly': bbox,
                    'text': text.strip(),
                    'confidence': confidence
                })
                
                # Debug: Print all detected text
                print(f"   📝 Detected: '{text.strip()}' (confidence: {confidence:.3f})")
        
        print(f"📝 Found {len(detections)} text regions")
        return detections
    
    def filter_serial_numbers(self, detections):
        """Filter detections to find fuel bundle serial numbers"""
        print("\n🔎 Analyzing text for serial number patterns...")
        print("-" * 50)
        
        serial_numbers = []
        
        for detection in detections:
            text = detection['text'].upper()
            # Remove any spaces or special characters
            clean_text = re.sub(r'[^A-Z0-9]', '', text)
            
            print(f"   🔍 Analyzing: '{text}' -> cleaned: '{clean_text}' (length: {len(clean_text)})")
            
            # Check different lengths to be more flexible
            for length in [6, 7, 8]:  # Allow 6-8 characters instead of just 7
                if len(clean_text) == length:
                    # For 6 characters, try to match patterns by removing one char
                    if length == 6:
                        # Try adding potential missing characters
                        test_texts = [clean_text + 'X', 'X' + clean_text]
                    elif length == 8:
                        # Try removing one character from different positions
                        test_texts = [clean_text[1:], clean_text[:-1]]
                    else:
                        test_texts = [clean_text]
                    
                    for test_text in test_texts:
                        if len(test_text) == 7:
                            # Check against our specific patterns
                            for i, pattern in enumerate(self.serial_patterns):
                                if re.match(pattern, test_text):
                                    pattern_name = ['AAnnnnA', 'AAnnnAn', 'AAAnnnA'][i]
                                    detection['serial_number'] = test_text
                                    detection['pattern'] = pattern_name
                                    detection['pattern_index'] = i + 1
                                    detection['original_text'] = text
                                    serial_numbers.append(detection)
                                    print(f"   ✅ FOUND SERIAL: '{test_text}' (Pattern {i+1}: {pattern_name})")
                                    break
                            if 'serial_number' in detection:
                                break
            
            # Additional flexible matching for common OCR mistakes
            if len(clean_text) >= 6 and len(clean_text) <= 8:
                # Common OCR substitutions
                ocr_fixes = {
                    '0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G', '2': 'Z'
                }
                
                # Try fixing common OCR errors
                for wrong, right in ocr_fixes.items():
                    fixed_text = clean_text.replace(wrong, right)
                    if len(fixed_text) == 7:
                        for i, pattern in enumerate(self.serial_patterns):
                            if re.match(pattern, fixed_text):
                                pattern_name = ['AAnnnnA', 'AAnnnAn', 'AAAnnnA'][i]
                                detection['serial_number'] = fixed_text
                                detection['pattern'] = pattern_name
                                detection['pattern_index'] = i + 1
                                detection['original_text'] = text
                                detection['ocr_corrected'] = True
                                if detection not in serial_numbers:
                                    serial_numbers.append(detection)
                                    print(f"   ✅ FOUND SERIAL (OCR corrected): '{fixed_text}' (Pattern {i+1}: {pattern_name})")
                                break
        
        print("-" * 50)
        return serial_numbers
    
    def draw_results(self, image, all_detections, serial_numbers):
        """Draw bounding boxes on the image"""
        result_image = image.copy()
        
        # Draw all text detections in blue (thin lines)
        for detection in all_detections:
            bbox = detection['bbox']
            cv2.rectangle(result_image, 
                         tuple(bbox[0]), tuple(bbox[1]), 
                         (100, 149, 237), 2)  # Cornflower blue, thicker for visibility
            
            # Add detected text label
            text_label = f"{detection['text']} ({detection['confidence']:.2f})"
            cv2.putText(result_image, text_label,
                       (bbox[0][0], bbox[0][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 149, 237), 1)
        
        # Draw serial numbers in bright red (thick lines)
        for sn in serial_numbers:
            bbox = sn['bbox']
            # Thick red rectangle
            cv2.rectangle(result_image, 
                         tuple(bbox[0]), tuple(bbox[1]), 
                         (255, 0, 0), 4)
            
            # Add label above the box
            label = f"SERIAL: {sn['serial_number']}"
            label_pos = (bbox[0][0], max(bbox[0][1] - 40, 30))
            
            # Add background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result_image,
                         (label_pos[0] - 5, label_pos[1] - text_height - 5),
                         (label_pos[0] + text_width + 5, label_pos[1] + baseline + 5),
                         (255, 0, 0), -1)
            
            # Add white text
            cv2.putText(result_image, label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add pattern info below the box
            pattern_label = f"Pattern: {sn['pattern']}"
            pattern_pos = (bbox[0][0], bbox[1][1] + 30)
            cv2.putText(result_image, pattern_label, pattern_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return result_image
    
    def detect_fuel_bundle_serial(self, image_path="image2.jpg"):
        """Main detection function"""
        print("🚀 Starting Fuel Bundle Serial Number Detection")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. Preprocess image
            image = self.preprocess_image(image_path)
            
            # 2. Detect all text
            all_detections = self.detect_text(image)
            
            # 3. Filter for serial numbers
            serial_numbers = self.filter_serial_numbers(all_detections)
            
            # 4. Create annotated image
            result_image = self.draw_results(image, all_detections, serial_numbers)
            
            # 5. Save result
            output_path = "detected_serial_number.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            
            # 6. Print results
            processing_time = time.time() - start_time
            self.print_results(serial_numbers, all_detections, processing_time, output_path)
            
            return serial_numbers
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def print_results(self, serial_numbers, all_detections, processing_time, output_path):
        """Print detection results to terminal"""
        print("\n" + "=" * 60)
        print("🎯 FUEL BUNDLE SERIAL NUMBER DETECTION RESULTS")
        print("=" * 60)
        
        # Show all detected text first
        print("📝 ALL DETECTED TEXT:")
        print("-" * 30)
        for i, detection in enumerate(all_detections, 1):
            print(f"{i}. '{detection['text']}' (confidence: {detection['confidence']:.3f})")
        print()
        
        if serial_numbers:
            print(f"✅ SUCCESS! Found {len(serial_numbers)} serial number(s):")
            print("-" * 40)
            
            for i, sn in enumerate(serial_numbers, 1):
                print(f"{i}. Serial Number: {sn['serial_number']}")
                print(f"   📊 Confidence: {sn['confidence']:.3f}")
                print(f"   📋 Pattern: {sn['pattern']}")
                print(f"   📍 Location: ({sn['bbox'][0][0]}, {sn['bbox'][0][1]}) to ({sn['bbox'][1][0]}, {sn['bbox'][1][1]})")
                if 'original_text' in sn:
                    print(f"   📄 Original OCR: '{sn['original_text']}'")
                if 'ocr_corrected' in sn:
                    print(f"   🔧 OCR Corrected: Yes")
                print()
        else:
            print("❌ No fuel bundle serial numbers detected!")
            print("\n🔍 Expected formats:")
            print("   1. AAnnnnA (e.g., FD8764S)")
            print("   2. AAnnnAn (e.g., GFD887E)")
            print("   3. AAAnnnA")
            
            if all_detections:
                print("\n💡 Detected text that didn't match patterns:")
                for detection in all_detections:
                    clean = re.sub(r'[^A-Z0-9]', '', detection['text'].upper())
                    print(f"   - '{detection['text']}' -> '{clean}' (length: {len(clean)})")
        
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"💾 Annotated image saved: {output_path}")
        print("=" * 60)

def main():
    """Main function"""
    print("🔧 Fuel Bundle Serial Number Detector v1.0")
    print("📋 Supported formats: AAnnnnA, AAnnnAn, AAAnnnA")
    print()
    
    # Check if image exists
    if not Path("image2.jpg").exists():
        print("❌ Error: image2.jpg not found in current directory!")
        print("Please place your fuel bundle image as 'image2.jpg'")
        return
    
    # Initialize detector
    detector = FuelBundleSerialDetector()
    
    # Run detection
    serial_numbers = detector.detect_fuel_bundle_serial("image2.jpg")
    
    # Return for programmatic use
    return serial_numbers

if __name__ == "__main__":
    main()