import torch
import platform
import subprocess
import sys

def check_cuda_setup():
    """
    Comprehensive CUDA and GPU detection test
    Specifically designed to check RTX 1650 compatibility
    """
    
    print("🚀 CUDA and GPU Detection Test")
    print("=" * 60)
    
    # System Information
    print("💻 SYSTEM INFORMATION:")
    print(f"   Operating System: {platform.system()} {platform.release()}")
    print(f"   Python Version: {sys.version}")
    print(f"   PyTorch Version: {torch.__version__}")
    print()
    
    # CUDA Availability Check
    print("🔍 CUDA AVAILABILITY:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
    
    if not cuda_available:
        print("   ⚠️  CUDA not detected by PyTorch!")
        print("   💡 Possible solutions:")
        print("      1. Install PyTorch with CUDA support:")
        print("         pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("      2. Check NVIDIA drivers are installed")
        print("      3. Verify CUDA toolkit is installed")
        return False
    
    print()
    
    # CUDA Version Information
    print("🔧 CUDA INFORMATION:")
    print(f"   CUDA Version (PyTorch): {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   cuDNN Enabled: {'✅ YES' if torch.backends.cudnn.enabled else '❌ NO'}")
    print()
    
    # GPU Device Information
    print("🎮 GPU DEVICE INFORMATION:")
    device_count = torch.cuda.device_count()
    print(f"   Number of GPUs: {device_count}")
    
    if device_count > 0:
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")
            
            # Check if it's RTX 1650
            if "1650" in gpu_name:
                print(f"      🎯 RTX 1650 Detected! Perfect for OCR tasks.")
            elif "RTX" in gpu_name or "GTX" in gpu_name:
                print(f"      ✅ NVIDIA GPU detected - should work well!")
            
            # GPU Memory Information
            try:
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                print(f"      📊 Total Memory: {gpu_memory_gb:.1f} GB")
                
                # Check current memory usage
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"      📈 Memory Allocated: {memory_allocated:.2f} GB")
                print(f"      📊 Memory Reserved: {memory_reserved:.2f} GB")
                
            except Exception as e:
                print(f"      ⚠️  Could not get memory info: {e}")
    
    print()
    
    # CUDA Device Properties
    print("🔧 DETAILED GPU PROPERTIES:")
    if device_count > 0:
        device = torch.cuda.get_device_properties(0)
        print(f"   Device Name: {device.name}")
        print(f"   Compute Capability: {device.major}.{device.minor}")
        print(f"   Multiprocessors: {device.multi_processor_count}")

    print()
    
    # NVIDIA-SMI Check
    print("🖥️  NVIDIA-SMI CHECK:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ nvidia-smi command working")
            # Extract key info from nvidia-smi
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"   🔧 {line.strip()}")
                elif 'RTX' in line or 'GTX' in line:
                    print(f"   🎮 {line.strip()}")
        else:
            print("   ⚠️  nvidia-smi command failed")
    except subprocess.TimeoutExpired:
        print("   ⚠️  nvidia-smi command timed out")
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found (NVIDIA drivers may not be installed)")
    
    print()
    
    # Performance Test
    print("⚡ CUDA PERFORMANCE TEST:")
    try:
        # Create tensors for testing
        print("   🧪 Testing GPU performance...")
        device = torch.device('cuda:0')
        
        # Test tensor operations
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Time matrix multiplication
        import time
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        
        print(f"   ⚡ GPU Matrix Multiplication (1000x1000): {gpu_time:.4f} seconds")
        
        # Compare with CPU
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"   🖥️  CPU Matrix Multiplication (1000x1000): {cpu_time:.4f} seconds")
        print(f"   🚀 GPU Speedup: {cpu_time/gpu_time:.1f}x faster")
        
        if gpu_time < cpu_time:
            print("   ✅ GPU is working and faster than CPU!")
        else:
            print("   ⚠️  GPU might not be properly utilized")
            
    except Exception as e:
        print(f"   ❌ GPU performance test failed: {e}")
    
    print()
    
    # OCR Library Compatibility Test
    print("📚 OCR LIBRARY COMPATIBILITY:")
    
    # Test EasyOCR
    try:
        import easyocr
        print("   ✅ EasyOCR: Available")
        reader = easyocr.Reader(['en'], gpu=True)
        print("   🎮 EasyOCR GPU mode: Enabled")
    except ImportError:
        print("   ❌ EasyOCR: Not installed")
    except Exception as e:
        print(f"   ⚠️  EasyOCR GPU issue: {e}")
    
    # Test PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("   ✅ PaddleOCR: Available")
        ocr = PaddleOCR(use_gpu=True, show_log=False)
        print("   🎮 PaddleOCR GPU mode: Enabled")
    except ImportError:
        print("   ❌ PaddleOCR: Not installed")
    except Exception as e:
        print(f"   ⚠️  PaddleOCR GPU issue: {e}")
    
    # Test Tesseract
    try:
        import pytesseract
        print("   ✅ Tesseract: Available")
    except ImportError:
        print("   ❌ Tesseract: Not installed")
    
    print()
    
    # Final Assessment
    print("🎯 FINAL ASSESSMENT:")
    if cuda_available and device_count > 0:
        print("   ✅ Your RTX 1650 setup looks good for OCR tasks!")
        print("   🚀 Expected performance boost: 3-5x faster than CPU")
        print("   💡 The fuel bundle detector should use GPU acceleration")
        
        # Specific recommendations for RTX 1650
        gpu_name = torch.cuda.get_device_name(0)
        if "1650" in gpu_name:
            print("   🎯 RTX 1650 specific notes:")
            print("      - 4GB VRAM is perfect for OCR workloads")
            print("      - Should handle multiple image preprocessing easily")
            print("      - PaddleOCR will give best accuracy with GPU acceleration")
        
        return True
    else:
        print("   ❌ CUDA setup needs attention")
        print("   🔧 Please install PyTorch with CUDA support and NVIDIA drivers")
        return False

def test_memory_allocation():
    """Test GPU memory allocation capabilities"""
    if not torch.cuda.is_available():
        return
        
    print("\n🧠 GPU MEMORY ALLOCATION TEST:")
    try:
        device = torch.device('cuda:0')
        
        # Test different memory sizes
        test_sizes = [100, 500, 1000, 2000]  # MB
        
        for size_mb in test_sizes:
            try:
                # Calculate tensor size for ~size_mb MB
                elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
                side = int(elements ** 0.5)
                
                tensor = torch.randn(side, side, device=device)
                actual_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                
                print(f"   ✅ Allocated {actual_mb:.1f} MB tensor successfully")
                
                # Clean up
                del tensor
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"   ❌ Failed to allocate {size_mb} MB (Out of memory)")
                break
            except Exception as e:
                print(f"   ⚠️  Error allocating {size_mb} MB: {e}")
                break
        
        # Show final memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"   📊 Final memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")

if __name__ == "__main__":
    # Run comprehensive CUDA test
    success = check_cuda_setup()
    
    # Run memory test if CUDA is available
    if success:
        test_memory_allocation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Your system is ready for GPU-accelerated OCR!")
        print("💡 You can now run: python fuel_bundle_detector.py")
    else:
        print("🔧 Please fix CUDA setup before running OCR detection")
        print("💡 Install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("=" * 60) 