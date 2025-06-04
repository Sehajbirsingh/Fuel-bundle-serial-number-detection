import torch
import cv2
import numpy as np
import re
from pathlib import Path
import time

# Try importing different OCR libraries
OCR_BACKENDS = {}

try:
    import easyocr
    OCR_BACKENDS['easyocr'] = True
except ImportError:
    OCR_BACKENDS['easyocr'] = False

try:
    from paddleocr import PaddleOCR
    OCR_BACKENDS['paddleocr'] = True
except ImportError:
    OCR_BACKENDS['paddleocr'] = False

try:
    import pytesseract
    OCR_BACKENDS['tesseract'] = True
except ImportError:
    OCR_BACKENDS['tesseract'] = False

class FuelBundleSerialDetector:
    """
    Advanced Fuel Bundle Serial Number Detector
    Supports multiple OCR backends: PaddleOCR, EasyOCR, Tesseract
    """
    
    def __init__(self, backend='auto'):
        print("🚀 Initializing Advanced Fuel Bundle Serial Number Detector...")
        
        # Auto-select best available backend
        if backend == 'auto':
            if OCR_BACKENDS['paddleocr']:
                backend = 'paddleocr'
            elif OCR_BACKENDS['easyocr']:
                backend = 'easyocr'
            elif OCR_BACKENDS['tesseract']:
                backend = 'tesseract'
            else:
                raise RuntimeError("No OCR backend available!")
        
        self.backend = backend
        print(f"🔧 Using OCR Backend: {backend.upper()}")
        
        # Initialize the selected backend
        self.setup_ocr_backend()
        
        # Define exact serial number patterns (7 characters)
        self.serial_patterns = [
            r'^[A-Z]{2}\d{4}[A-Z]$',  # AAnnnnA (FD8764S, YUH118S)
            r'^[A-Z]{2}\d{3}[A-Z]\d$', # AAnnnAn (GFD887E) 
            r'^[A-Z]{3}\d{3}[A-Z]$'   # AAAnnnA
        ]
        
        print("✅ Advanced OCR system loaded successfully!")
    
    def setup_ocr_backend(self):
        """Setup the selected OCR backend"""
        if self.backend == 'paddleocr':
            print("🎯 Loading PaddleOCR (Best Accuracy)...")
            self.ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en',
                use_gpu=torch.cuda.is_available(),
                show_log=False
            )
            
        elif self.backend == 'easyocr':
            print("🎯 Loading EasyOCR...")
            self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
        elif self.backend == 'tesseract':
            print("🎯 Loading Tesseract...")
            # Configure Tesseract for better accuracy
            self.tesseract_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    def preprocess_image(self, image_path):
        """Enhanced preprocessing for fuel bundle images"""
        print(f"📸 Loading image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"❌ Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Multiple preprocessing versions for better OCR
        processed_versions = []
        
        # Version 1: Original with contrast enhancement
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        processed_versions.append(('enhanced', enhanced_rgb))
        
        # Version 2: Grayscale with adaptive threshold
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        processed_versions.append(('threshold', thresh_rgb))
        
        # Version 3: High contrast
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Apply bilateral filter to smooth the image while keeping edges sharp
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        # Apply threshold
        _, binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        processed_versions.append(('binary', binary_rgb))
        
        print("✅ Created multiple preprocessed versions for better OCR")
        return processed_versions
    
    def detect_text_paddleocr(self, image_versions):
        """Detect text using PaddleOCR"""
        print("🔍 Detecting text with PaddleOCR...")
        
        all_detections = []
        
        for version_name, image in image_versions:
            print(f"   📝 Processing {version_name} version...")
            try:
                results = self.ocr.ocr(image, cls=True)
                
                if results and results[0]:
                    for line in results[0]:
                        bbox, (text, confidence) = line
                        if confidence > 0.5:  # PaddleOCR confidence threshold
                            # Convert bbox format
                            bbox_np = np.array(bbox, dtype=np.int32)
                            x_min, y_min = bbox_np[:, 0].min(), bbox_np[:, 1].min()
                            x_max, y_max = bbox_np[:, 0].max(), bbox_np[:, 1].max()
                            
                            detection = {
                                'bbox': [[x_min, y_min], [x_max, y_max]],
                                'bbox_poly': bbox,
                                'text': text.strip(),
                                'confidence': confidence,
                                'version': version_name
                            }
                            all_detections.append(detection)
                            print(f"      🎯 Found: '{text.strip()}' (conf: {confidence:.3f})")
                            
            except Exception as e:
                print(f"      ⚠️ Error processing {version_name}: {e}")
                continue
        
        return all_detections
    
    def detect_text_easyocr(self, image_versions):
        """Detect text using EasyOCR"""
        print("🔍 Detecting text with EasyOCR...")
        
        all_detections = []
        
        for version_name, image in image_versions:
            print(f"   📝 Processing {version_name} version...")
            try:
                results = self.ocr.readtext(image, detail=1, paragraph=False)
                
                for bbox, text, confidence in results:
                    if confidence > 0.3:
                        bbox_np = np.array(bbox, dtype=np.int32)
                        x_min, y_min = bbox_np[:, 0].min(), bbox_np[:, 1].min()
                        x_max, y_max = bbox_np[:, 0].max(), bbox_np[:, 1].max()
                        
                        detection = {
                            'bbox': [[x_min, y_min], [x_max, y_max]],
                            'bbox_poly': bbox,
                            'text': text.strip(),
                            'confidence': confidence,
                            'version': version_name
                        }
                        all_detections.append(detection)
                        print(f"      🎯 Found: '{text.strip()}' (conf: {confidence:.3f})")
                        
            except Exception as e:
                print(f"      ⚠️ Error processing {version_name}: {e}")
                continue
        
        return all_detections
    
    def detect_text_tesseract(self, image_versions):
        """Detect text using Tesseract"""
        print("🔍 Detecting text with Tesseract...")
        
        all_detections = []
        
        for version_name, image in image_versions:
            print(f"   📝 Processing {version_name} version...")
            try:
                # Convert to format Tesseract expects
                if len(image.shape) == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = image
                
                # Get text and bounding boxes
                data = pytesseract.image_to_data(image_gray, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
                
                for i, text in enumerate(data['text']):
                    confidence = int(data['conf'][i])
                    if confidence > 30 and text.strip():  # Tesseract confidence is 0-100
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        detection = {
                            'bbox': [[x, y], [x + w, y + h]],
                            'text': text.strip(),
                            'confidence': confidence / 100.0,  # Normalize to 0-1
                            'version': version_name
                        }
                        all_detections.append(detection)
                        print(f"      🎯 Found: '{text.strip()}' (conf: {confidence/100:.3f})")
                        
            except Exception as e:
                print(f"      ⚠️ Error processing {version_name}: {e}")
                continue
        
        return all_detections
    
    def detect_text(self, image_versions):
        """Detect text using the selected backend"""
        if self.backend == 'paddleocr':
            return self.detect_text_paddleocr(image_versions)
        elif self.backend == 'easyocr':
            return self.detect_text_easyocr(image_versions)
        elif self.backend == 'tesseract':
            return self.detect_text_tesseract(image_versions)
    
    def filter_serial_numbers(self, detections):
        """Enhanced serial number filtering with fuzzy matching"""
        print("\n🔎 Advanced serial number pattern analysis...")
        print("-" * 60)
        
        serial_numbers = []
        
        for detection in detections:
            text = detection['text'].upper()
            clean_text = re.sub(r'[^A-Z0-9]', '', text)
            
            print(f"   🔍 '{text}' -> '{clean_text}' (len: {len(clean_text)}, ver: {detection.get('version', 'unknown')})")
            
            # Check exact match first
            if len(clean_text) == 7:
                for i, pattern in enumerate(self.serial_patterns):
                    if re.match(pattern, clean_text):
                        pattern_name = ['AAnnnnA', 'AAnnnAn', 'AAAnnnA'][i]
                        detection['serial_number'] = clean_text
                        detection['pattern'] = pattern_name
                        detection['match_type'] = 'exact'
                        serial_numbers.append(detection)
                        print(f"      ✅ EXACT MATCH: '{clean_text}' (Pattern: {pattern_name})")
                        break
            
            # Fuzzy matching for OCR errors
            if 'serial_number' not in detection and 6 <= len(clean_text) <= 8:
                # Common OCR error corrections
                corrections = {
                    '0': ['O', 'Q'], '1': ['I', 'L'], '2': ['Z'], '5': ['S'], 
                    '6': ['G'], '8': ['B'], 'O': ['0'], 'I': ['1'], 'S': ['5'],
                    'B': ['8'], 'G': ['6'], 'Z': ['2'], 'Q': ['0']
                }
                
                # Generate possible corrections
                possible_texts = [clean_text]
                for orig, replacements in corrections.items():
                    for replacement in replacements:
                        if orig in clean_text:
                            possible_texts.append(clean_text.replace(orig, replacement))
                
                # Test all possibilities
                for possible_text in possible_texts:
                    if len(possible_text) == 7:
                        for i, pattern in enumerate(self.serial_patterns):
                            if re.match(pattern, possible_text):
                                pattern_name = ['AAnnnnA', 'AAnnnAn', 'AAAnnnA'][i]
                                detection['serial_number'] = possible_text
                                detection['pattern'] = pattern_name
                                detection['match_type'] = 'corrected'
                                detection['original_ocr'] = clean_text
                                serial_numbers.append(detection)
                                print(f"      ✅ CORRECTED MATCH: '{clean_text}' -> '{possible_text}' (Pattern: {pattern_name})")
                                break
                    if 'serial_number' in detection:
                        break
        
        # Remove duplicates (same serial number from different preprocessing versions)
        unique_serials = []
        seen_serials = set()
        for sn in serial_numbers:
            if sn['serial_number'] not in seen_serials:
                unique_serials.append(sn)
                seen_serials.add(sn['serial_number'])
        
        print("-" * 60)
        return unique_serials
    
    def draw_results(self, image, all_detections, serial_numbers):
        """Enhanced visualization"""
        # Use the first (enhanced) version for display
        result_image = image.copy()
        
        # Draw all detections in blue
        for detection in all_detections:
            bbox = detection['bbox']
            cv2.rectangle(result_image, tuple(bbox[0]), tuple(bbox[1]), (100, 149, 237), 2)
            
            # Add text label
            label = f"{detection['text']} ({detection['confidence']:.2f})"
            cv2.putText(result_image, label, (bbox[0][0], bbox[0][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 149, 237), 1)
        
        # Draw serial numbers in red
        for sn in serial_numbers:
            bbox = sn['bbox']
            cv2.rectangle(result_image, tuple(bbox[0]), tuple(bbox[1]), (255, 0, 0), 4)
            
            # Large serial number label
            label = f"SERIAL: {sn['serial_number']}"
            label_pos = (bbox[0][0], max(bbox[0][1] - 50, 40))
            
            # Background for label
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(result_image,
                         (label_pos[0] - 5, label_pos[1] - text_height - 5),
                         (label_pos[0] + text_width + 5, label_pos[1] + baseline + 5),
                         (255, 0, 0), -1)
            
            # White text
            cv2.putText(result_image, label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Pattern and method info
            info = f"{sn['pattern']} | {sn.get('match_type', 'exact').upper()}"
            cv2.putText(result_image, info, (bbox[0][0], bbox[1][1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return result_image
    
    def detect_fuel_bundle_serial(self, image_path="image.png"):
        """Main detection function with advanced OCR"""
        print("🚀 Starting Advanced Fuel Bundle Serial Detection")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # 1. Preprocess image (multiple versions)
            image_versions = self.preprocess_image(image_path)
            
            # 2. Detect text with selected backend
            all_detections = self.detect_text(image_versions)
            
            # 3. Filter for serial numbers
            serial_numbers = self.filter_serial_numbers(all_detections)
            
            # 4. Create result image (use enhanced version)
            result_image = self.draw_results(image_versions[0][1], all_detections, serial_numbers)
            
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
        """Print comprehensive results"""
        print("\n" + "=" * 70)
        print("🎯 ADVANCED FUEL BUNDLE SERIAL DETECTION RESULTS")
        print("=" * 70)
        
        print(f"🔧 OCR Backend: {self.backend.upper()}")
        print(f"📝 Total detections: {len(all_detections)}")
        
        if all_detections:
            print("\n📋 ALL DETECTED TEXT:")
            print("-" * 40)
            for i, det in enumerate(all_detections, 1):
                version = det.get('version', 'unknown')
                print(f"{i}. '{det['text']}' (conf: {det['confidence']:.3f}, ver: {version})")
        
        if serial_numbers:
            print(f"\n✅ SUCCESS! Found {len(serial_numbers)} serial number(s):")
            print("-" * 50)
            
            for i, sn in enumerate(serial_numbers, 1):
                print(f"\n{i}. 🎯 SERIAL NUMBER: {sn['serial_number']}")
                print(f"   📊 Confidence: {sn['confidence']:.3f}")
                print(f"   📋 Pattern: {sn['pattern']}")
                print(f"   🔧 Match Type: {sn.get('match_type', 'exact').upper()}")
                print(f"   📍 Location: ({sn['bbox'][0][0]}, {sn['bbox'][0][1]}) to ({sn['bbox'][1][0]}, {sn['bbox'][1][1]})")
                if 'original_ocr' in sn:
                    print(f"   📝 Original OCR: '{sn['original_ocr']}'")
                if 'version' in sn:
                    print(f"   🖼️  Best Version: {sn['version']}")
        else:
            print("\n❌ No fuel bundle serial numbers detected!")
            print("\n🔍 Expected formats: AAnnnnA, AAnnnAn, AAAnnnA")
        
        print(f"\n⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"💾 Result saved: {output_path}")
        print("=" * 70)

def main():
    """Main function with backend selection"""
    print("🔧 Advanced Fuel Bundle Serial Number Detector v2.0")
    print("=" * 60)
    
    # Show available backends
    print("📚 Available OCR Backends:")
    for backend, available in OCR_BACKENDS.items():
        status = "✅ Available" if available else "❌ Not installed"
        print(f"   {backend.upper()}: {status}")
    print()
    
    if not Path("image.png").exists():
        print("❌ Error: image.png not found!")
        return
    
    # Try PaddleOCR first (most accurate), fallback to others
    detector = FuelBundleSerialDetector(backend='auto')
    serial_numbers = detector.detect_fuel_bundle_serial("image.png")
    
    return serial_numbers

if __name__ == "__main__":
    main()