import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse

class MetalSerialDetector:
    """
    Specialized detector for metal engraved serial numbers on fuel bundles
    Focuses on bounding box detection for challenging metallic surfaces
    """
    
    def __init__(self):
        print("🔧 Initializing Metal Engraved Serial Number Detector")
        print("🎯 Specialized for fuel bundle metallic surfaces")
        print("=" * 60)
    
    def load_image(self, image_path):
        """Load and validate image"""
        print(f"📸 Loading image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"❌ Could not load image: {image_path}")
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✅ Image loaded: {image_rgb.shape[1]}x{image_rgb.shape[0]} pixels")
        
        return image_rgb
    
    def preprocess_for_metal_engraving(self, image):
        """
        Advanced preprocessing specifically for metal engraved text
        Handles reflective surfaces, shadows, and embossed text
        """
        print("🔧 Applying metal engraving preprocessing...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create multiple preprocessing versions for metal engravings
        processed_versions = []
        
        # Version 1: Enhanced contrast for metallic surfaces
        print("   📝 Creating enhanced contrast version...")
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_versions.append(('enhanced_contrast', enhanced))
        
        # Version 2: Gradient-based edge detection for embossed text
        print("   📝 Creating gradient edge version...")
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_norm = np.uint8(255 * gradient_magnitude / gradient_magnitude.max())
        processed_versions.append(('gradient_edges', gradient_norm))
        
        # Version 3: Morphological operations for engraved text
        print("   📝 Creating morphological version...")
        # Apply morphological operations to enhance text structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
        processed_versions.append(('morphological', morph))
        
        # Version 4: Adaptive threshold for varying lighting
        print("   📝 Creating adaptive threshold version...")
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        processed_versions.append(('adaptive_thresh', adaptive))
        
        # Version 5: Unsharp masking for engraved detail enhancement
        print("   📝 Creating unsharp mask version...")
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        processed_versions.append(('unsharp_mask', unsharp))
        
        # Version 6: High-pass filter for embossed text
        print("   📝 Creating high-pass filter version...")
        lowpass = cv2.GaussianBlur(gray, (21, 21), 0)
        highpass = cv2.subtract(gray, lowpass)
        highpass_enhanced = cv2.add(highpass, 127)  # Shift to visible range
        processed_versions.append(('high_pass', highpass_enhanced))
        
        print(f"✅ Created {len(processed_versions)} preprocessing versions")
        return processed_versions
    
    def detect_text_regions_mser(self, image, version_name):
        """Use MSER (Maximally Stable Extremal Regions) for text detection"""
        try:
            mser = cv2.MSER_create(
                _min_area=50,     # Minimum area of text regions
                _max_area=2000,   # Maximum area of text regions
                _delta=5          # Stability threshold
            )
            
            regions, _ = mser.detectRegions(image)
            
            bboxes = []
            for region in regions:
                # Get bounding rectangle for each region
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                
                # Filter by aspect ratio (typical for text)
                aspect_ratio = w / h if h > 0 else 0
                if 1.5 <= aspect_ratio <= 8.0 and w >= 30 and h >= 8:
                    bboxes.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': w * h,
                        'aspect_ratio': aspect_ratio,
                        'method': 'MSER',
                        'version': version_name
                    })
            
            return bboxes
            
        except Exception as e:
            print(f"      ⚠️ MSER failed on {version_name}: {e}")
            return []
    
    def detect_text_regions_contours(self, image, version_name):
        """Use contour detection for text regions"""
        try:
            # Apply threshold if image is grayscale
            if len(image.shape) == 2:
                _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bboxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter by size and aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if (area > 100 and 1.5 <= aspect_ratio <= 10.0 and 
                    w >= 25 and h >= 6 and w <= 300 and h <= 50):
                    
                    bboxes.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'method': 'Contours',
                        'version': version_name
                    })
            
            return bboxes
            
        except Exception as e:
            print(f"      ⚠️ Contour detection failed on {version_name}: {e}")
            return []
    
    def detect_text_regions_connected_components(self, image, version_name):
        """Use connected components for text detection"""
        try:
            # Ensure binary image
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Connected components analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            
            bboxes = []
            for i in range(1, num_labels):  # Skip background (label 0)
                x, y, w, h, area = stats[i]
                
                aspect_ratio = w / h if h > 0 else 0
                if (area > 80 and 1.2 <= aspect_ratio <= 12.0 and 
                    w >= 20 and h >= 5 and w <= 400 and h <= 60):
                    
                    bboxes.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'method': 'Connected Components',
                        'version': version_name
                    })
            
            return bboxes
            
        except Exception as e:
            print(f"      ⚠️ Connected components failed on {version_name}: {e}")
            return []
    
    def detect_all_text_regions(self, processed_versions):
        """Apply multiple detection methods to all preprocessing versions"""
        print("\n🔍 Detecting text regions using multiple methods...")
        
        all_detections = []
        
        for version_name, image in processed_versions:
            print(f"   📝 Processing {version_name} version...")
            
            # Method 1: MSER
            mser_boxes = self.detect_text_regions_mser(image, version_name)
            all_detections.extend(mser_boxes)
            if mser_boxes:
                print(f"      🎯 MSER found {len(mser_boxes)} regions")
            
            # Method 2: Contours
            contour_boxes = self.detect_text_regions_contours(image, version_name)
            all_detections.extend(contour_boxes)
            if contour_boxes:
                print(f"      🎯 Contours found {len(contour_boxes)} regions")
            
            # Method 3: Connected Components
            cc_boxes = self.detect_text_regions_connected_components(image, version_name)
            all_detections.extend(cc_boxes)
            if cc_boxes:
                print(f"      🎯 Connected Components found {len(cc_boxes)} regions")
        
        print(f"\n📊 Total detections before filtering: {len(all_detections)}")
        return all_detections
    
    def filter_and_merge_detections(self, detections):
        """Filter and merge overlapping detections"""
        print("\n🔧 Filtering and merging detections...")
        
        if not detections:
            return []
        
        # Sort by area (larger regions first)
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        # Remove overlapping detections (Non-Maximum Suppression)
        filtered = []
        for detection in detections:
            is_duplicate = False
            
            for existing in filtered:
                # Calculate overlap
                x1, y1, w1, h1 = existing['x'], existing['y'], existing['w'], existing['h']
                x2, y2, w2, h2 = detection['x'], detection['y'], detection['w'], detection['h']
                
                # Calculate intersection
                left = max(x1, x2)
                top = max(y1, y2)
                right = min(x1 + w1, x2 + w2)
                bottom = min(y1 + h1, y2 + h2)
                
                if left < right and top < bottom:
                    intersection_area = (right - left) * (bottom - top)
                    union_area = w1 * h1 + w2 * h2 - intersection_area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > 0.3:  # 30% overlap threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        # Additional filtering for serial number characteristics
        final_filtered = []
        for detection in filtered:
            # Filter by typical serial number dimensions and aspect ratios
            w, h = detection['w'], detection['h']
            aspect_ratio = detection['aspect_ratio']
            
            # Typical serial number characteristics
            if (w >= 40 and h >= 8 and w <= 250 and h <= 40 and 
                2.0 <= aspect_ratio <= 8.0):
                final_filtered.append(detection)
        
        print(f"📊 Filtered from {len(detections)} to {len(final_filtered)} detections")
        return final_filtered
    
    def visualize_detections(self, image, detections, save_path):
        """Create visualization with bounding boxes"""
        print(f"\n🎨 Creating visualization with {len(detections)} detections...")
        
        # Create a copy for drawing
        vis_image = image.copy()
        
        # Color map for different methods
        method_colors = {
            'MSER': (255, 0, 0),           # Red
            'Contours': (0, 255, 0),       # Green
            'Connected Components': (0, 0, 255)  # Blue
        }
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            method = detection['method']
            version = detection['version']
            
            # Get color for method
            color = method_colors.get(method, (255, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label with detection info
            label = f"#{i+1} {method}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for label
            cv2.rectangle(vis_image, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 5, y - 5), 
                         color, -1)
            
            # White text
            cv2.putText(vis_image, label, (x + 2, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add aspect ratio info below box
            info = f"AR:{detection['aspect_ratio']:.1f} A:{detection['area']}"
            cv2.putText(vis_image, info, (x, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add legend
        legend_y = 30
        for method, color in method_colors.items():
            cv2.rectangle(vis_image, (10, legend_y), (30, legend_y + 15), color, -1)
            cv2.putText(vis_image, method, (35, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 25
        
        # Save visualization
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"💾 Visualization saved: {save_path}")
        
        return vis_image
    
    def detect_metal_serial_numbers(self, image_path="image.png", output_dir="metal_detection_output"):
        """Main detection pipeline for metal engraved serial numbers"""
        print("🚀 Starting Metal Serial Number Detection")
        print("🎯 Specialized for fuel bundle metal engravings")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # 1. Load image
            image = self.load_image(image_path)
            
            # 2. Preprocess for metal engravings
            processed_versions = self.preprocess_for_metal_engraving(image)
            
            # 3. Detect text regions using multiple methods
            all_detections = self.detect_all_text_regions(processed_versions)
            
            # 4. Filter and merge detections
            final_detections = self.filter_and_merge_detections(all_detections)
            
            # 5. Create visualization
            input_name = Path(image_path).stem
            output_path = Path(output_dir) / f"{input_name}_metal_detections.jpg"
            vis_image = self.visualize_detections(image, final_detections, str(output_path))
            
            # 6. Print results
            processing_time = time.time() - start_time
            self.print_results(final_detections, processing_time, str(output_path))
            
            return final_detections, str(output_path)
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return [], ""
    
    def print_results(self, detections, processing_time, output_path):
        """Print detailed detection results"""
        print("\n" + "=" * 70)
        print("🎯 METAL SERIAL NUMBER DETECTION RESULTS")
        print("=" * 70)
        
        if detections:
            print(f"✅ SUCCESS! Found {len(detections)} potential serial number regions:")
            print("-" * 50)
            
            for i, detection in enumerate(detections, 1):
                print(f"\n{i}. 📍 DETECTION #{i}")
                print(f"   🔧 Method: {detection['method']}")
                print(f"   🖼️  Source: {detection['version']}")
                print(f"   📊 Location: ({detection['x']}, {detection['y']}) - {detection['w']}x{detection['h']}")
                print(f"   📏 Aspect Ratio: {detection['aspect_ratio']:.2f}")
                print(f"   📐 Area: {detection['area']} pixels")
        else:
            print("❌ No serial number regions detected!")
            print("\n💡 Suggestions:")
            print("   - Check if the image has good lighting")
            print("   - Ensure serial numbers are clearly visible")
            print("   - Try adjusting the image angle/distance")
        
        print(f"\n⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"💾 Visualization saved: {output_path}")
        print("=" * 70)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Metal Serial Number Detector for Fuel Bundles')
    parser.add_argument('--image', type=str, default='image.png',
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='metal_detection_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🔧 Metal Engraved Serial Number Detector v1.0")
    print("🎯 Specialized for fuel bundle metallic surfaces")
    print("=" * 60)
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Error: {args.image} not found!")
        print("Please provide a valid image path")
        return
    
    # Initialize detector
    detector = MetalSerialDetector()
    
    # Run detection
    detections, output_path = detector.detect_metal_serial_numbers(args.image, args.output)
    
    # Summary
    if detections:
        print(f"\n🎉 Detection complete! Found {len(detections)} potential serial numbers")
        print(f"📁 Check results in: {output_path}")
    else:
        print("\n🔍 No serial numbers detected. Try adjusting lighting or image angle.")
    
    return detections

if __name__ == "__main__":
    main() 