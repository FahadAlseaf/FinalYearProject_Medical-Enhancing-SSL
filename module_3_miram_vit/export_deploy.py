"""
export_deploy.py - ONNX Export and Speed Benchmarking

This script exports the trained MIRAM model to ONNX format for
deployment and benchmarks inference speed.

Key Features:
    - ONNX export with dynamic batch size
    - PyTorch vs ONNX Runtime speed comparison
    - FPS calculation for real-time assessment

Usage:
    python export_deploy.py

Output:
    - miram_model_optimized.onnx: Exported model
    - Benchmark results printed to console

Requirements:
    pip install onnx onnxruntime-gpu  # or onnxruntime for CPU
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np

from config import DEVICE, BEST_MODEL_PATH, MODEL_DIR, IMG_SIZE
from models import MIRAM


# ===========================================
# INFERENCE WRAPPER
# ===========================================

class MIRAMInferenceWrapper(nn.Module):
    """
    Wrapper for ONNX export that fixes mask_ratio=0 (inference mode).
    
    In inference/denoising mode, we don't mask any patches.
    This wrapper simplifies the model interface for deployment.
    """
    
    def __init__(self, model: MIRAM):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with no masking.
        
        Args:
            x: Input images (N, 1, H, W)
            
        Returns:
            pred_fine: Restored patch predictions (N, L, patch_size^2)
        """
        pred_fine, _, _ = self.model(x, mask_ratio=0.0)
        return pred_fine


# ===========================================
# ONNX EXPORT
# ===========================================

def export_to_onnx():
    """Export MIRAM model to ONNX format with benchmarking."""
    
    print("=" * 60)
    print("MIRAM ONNX Export & Benchmarking")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load PyTorch model
    device = torch.device(DEVICE)
    model = MIRAM().to(device)
    
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        )
        print(f"✅ Loaded PyTorch model from {BEST_MODEL_PATH}")
    else:
        print(f"⚠️ Model weights not found at {BEST_MODEL_PATH}")
        print("   Exporting untrained model for testing...")
    
    model.eval()
    
    # Wrap model for inference
    wrapped_model = MIRAMInferenceWrapper(model)
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    
    # Export to ONNX
    onnx_path = os.path.join(MODEL_DIR, "miram_model_optimized.onnx")
    
    print(f"\n📦 Exporting to ONNX format...")
    print(f"   Input shape: {dummy_input.shape}")
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['restored_patches'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'restored_patches': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Export successful: {onnx_path}")
    
    # Get file size
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"   Model size: {file_size_mb:.2f} MB")
    
    # ==========================================
    # BENCHMARKING
    # ==========================================
    
    print("\n⏱️ RUNNING SPEED BENCHMARK")
    print("-" * 50)
    
    num_iterations = 50
    
    # PyTorch Benchmark
    print(f"Testing PyTorch latency ({num_iterations} iterations)...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = wrapped_model(dummy_input)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = wrapped_model(dummy_input)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    
    pt_time = (time.time() - start) / num_iterations
    
    # ONNX Runtime Benchmark
    print(f"Testing ONNX Runtime latency ({num_iterations} iterations)...")
    
    try:
        import onnxruntime as ort
        
        # Select execution provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        
        print(f"   Using providers: {providers}")
        
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Prepare input
        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            return tensor.cpu().numpy()
        
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(dummy_input)
        }
        
        # Warmup
        for _ in range(5):
            _ = ort_session.run(None, ort_inputs)
        
        start = time.time()
        for _ in range(num_iterations):
            _ = ort_session.run(None, ort_inputs)
        onnx_time = (time.time() - start) / num_iterations
        
        onnx_available = True
        
    except ImportError:
        print("   ⚠️ onnxruntime not installed. Skipping ONNX benchmark.")
        print("   Install with: pip install onnxruntime-gpu")
        onnx_time = None
        onnx_available = False
    except Exception as e:
        print(f"   ⚠️ ONNX Runtime error: {e}")
        onnx_time = None
        onnx_available = False
    
    # Print Results
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    pt_fps = 1.0 / pt_time if pt_time > 0 else 0
    print(f"PyTorch Inference:  {pt_time*1000:.2f} ms/image ({pt_fps:.1f} FPS)")
    
    if onnx_available and onnx_time is not None:
        onnx_fps = 1.0 / onnx_time if onnx_time > 0 else 0
        print(f"ONNX Optimized:     {onnx_time*1000:.2f} ms/image ({onnx_fps:.1f} FPS)")
        
        speedup = pt_time / onnx_time if onnx_time > 0 else 0
        print(f"\n🚀 Speedup Factor: {speedup:.2f}x")
        
        # Check requirement
        target_time = 0.5  # 500ms target
        if onnx_time < target_time:
            print(f"\n✅ REQUIREMENT MET: Inference < {target_time*1000:.0f}ms")
        else:
            print(f"\n⚠️ WARNING: Inference ({onnx_time*1000:.0f}ms) > target ({target_time*1000:.0f}ms)")
    
    print("=" * 60)
    
    return onnx_path


# ===========================================
# ONNX INFERENCE FUNCTION
# ===========================================

def run_onnx_inference(onnx_path: str, image_tensor: torch.Tensor) -> np.ndarray:
    """
    Run inference using ONNX Runtime.
    
    Args:
        onnx_path: Path to ONNX model file
        image_tensor: Input tensor (N, 1, H, W)
        
    Returns:
        Output predictions as numpy array
    """
    import onnxruntime as ort
    
    session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    input_name = session.get_inputs()[0].name
    
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.cpu().numpy()
    else:
        image_np = image_tensor
    
    outputs = session.run(None, {input_name: image_np})
    
    return outputs[0]


# ===========================================
# MAIN
# ===========================================

if __name__ == "__main__":
    # Check for onnxruntime
    try:
        import onnxruntime
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError:
        print("Installing onnxruntime...")
        import subprocess
        subprocess.run(['pip', 'install', 'onnx', 'onnxruntime'], check=True)
    
    export_to_onnx()
