"""
Train YOLO v8 Character Detection Model
Uses the created dataset with 167 images (129 train, 38 val)
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def train_character_detection_model():
    """Train YOLO v8 for character-level detection"""
    
    print("🚀 Starting YOLO v8 Character Detection Training")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎯 Using device: {device}")
    
    # Initialize YOLO model
    print("📥 Loading YOLO v8 nano model...")
    model = YOLO('yolov8n.pt')  # nano model for faster training
    
    # Check if dataset exists
    dataset_path = Path('yolo_character_dataset/dataset.yaml')
    if not dataset_path.exists():
        print("❌ Dataset not found! Run filename_based_character_trainer.py first")
        return
    
    print(f"📊 Dataset found: {dataset_path}")
    print("📊 Training on 167 images (129 train, 38 val)")
    print("📊 36 character classes (A-Z, 0-9)")
    
    # Train the model
    print("\n🎯 Starting training...")
    results = model.train(
        data=str(dataset_path),
        epochs=50,              
        imgsz=640,             
        batch=16,               
        name='fuel_character_detection',  
        patience=15,           
        save_period=10,        
        device=device,         
        workers=4,             
        project='runs/detect', 
        exist_ok=True,         
        pretrained=True,       
        optimizer='AdamW',     
        lr0=0.01,              
        momentum=0.937,        
        weight_decay=0.0005,   
        cls=0.5,               
        box=7.5,               
        dfl=1.5,               
        plots=True,            
        save=True,             
        verbose=True           
    )
    
    print("\n✅ Training Complete!")
    print(f"📁 Results saved to: runs/detect/fuel_character_detection/")
    print(f"🎯 Best model: runs/detect/fuel_character_detection/weights/best.pt")
    print(f"🎯 Last model: runs/detect/fuel_character_detection/weights/last.pt")
    
    return results

def test_trained_model(model_path="runs/detect/fuel_character_detection/weights/best.pt", 
                      test_image="test_images/AE6813U.jpg"):
    """Test the trained model on a sample image"""
    
    print("\n🔍 Testing Trained Model")
    print("=" * 40)
    
    # Load trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model(test_image)
    
    # Process results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"📸 Image: {test_image}")
            print(f"🔍 Found {len(boxes)} characters:")
            
            detected_chars = []
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # Character classes mapping
                classes = {
                    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
                    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
                    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
                    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2',
                    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
                }
                
                char = classes.get(class_id, 'UNKNOWN')
                detected_chars.append((x1, char, confidence))
                
                print(f"  {i+1}. Character: {char}, Confidence: {confidence:.2f}, Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            
            # Sort by x-coordinate and reconstruct serial number
            detected_chars.sort(key=lambda x: x[0])
            reconstructed = ''.join([char for _, char, _ in detected_chars])
            print(f"🎯 Reconstructed Serial: {reconstructed}")
            
            # Save annotated image
            annotated = result.plot()
            import cv2
            cv2.imwrite(f"test_result_{Path(test_image).stem}.jpg", annotated)
            print(f"💾 Annotated image saved: test_result_{Path(test_image).stem}.jpg")
        else:
            print("❌ No characters detected")
    
    return results

def main():
    """Main training and testing pipeline"""
    
    # Train the model
    results = train_character_detection_model()
    
    # Test the trained model
    print("\n" + "="*60)
    test_trained_model()
    
    print("\n🎯 Training Complete! Next Steps:")
    print("1. Check training results in 'runs/detect/fuel_character_detection/'")
    print("2. View training plots and metrics")
    print("3. Test on more images using the trained model")
    print("4. Deploy for production use")

if __name__ == "__main__":
    main()
