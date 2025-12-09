# export_deploy.py - ONNX EXPORT & SPEED BENCHMARK
import torch
import torch.onnx
import time
import numpy as np
import onnxruntime as ort
import os
from config import *
from models import MIRAM

def export_to_onnx():
    print("🚀 STARTING MODEL OPTIMIZATION & EXPORT")
    print("="*50)
    
    # 1. Load PyTorch Model
    device = torch.device(DEVICE)
    model = MIRAM().to(device)
    
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"✅ Loaded PyTorch model from {BEST_MODEL_PATH}")
    else:
        print("❌ Model weights not found!")
        return

    model.eval()
    
    # 2. Create Dummy Input (Standard MRI Size)
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    
    # 3. Export to ONNX
    onnx_path = os.path.join(MODEL_DIR, "miram_model_optimized.onnx")
    
    print(f"\n📦 Exporting to ONNX format...")
    # We export the Encoder + Fine Decoder path (Inference Mode)
    # Note: We need to wrap the model to handle the mask_ratio argument during export
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            # Force mask_ratio=0 for inference (Denoising mode)
            pred_fine, _, _ = self.model(x, mask_ratio=0.0)
            return pred_fine

    wrapped_model = InferenceWrapper(model)
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['restored_patches'],
        dynamic_axes={'input_image': {0: 'batch_size'}, 'restored_patches': {0: 'batch_size'}}
    )
    print(f"✅ Export successful: {onnx_path}")
    
    # ==========================================
    # 4. BENCHMARKING (Thesis Requirement)
    # ==========================================
    print("\n⏱️ RUNNING SPEED BENCHMARK (PyTorch vs ONNX)")
    print("-" * 50)
    
    # PyTorch Benchmark
    print("Testing PyTorch Latency...")
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = wrapped_model(dummy_input)
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    pt_time = (time.time() - start) / 50
    
    # ONNX Benchmark
    print("Testing ONNX Runtime Latency...")
    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Helper to unpatchify in numpy (since ONNX output is patches)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    
    start = time.time()
    for _ in range(50):
        _ = ort_session.run(None, ort_inputs)
    onnx_time = (time.time() - start) / 50
    
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 50)
    print(f"PyTorch Inference: {pt_time*1000:.2f} ms/image ({1/pt_time:.1f} FPS)")
    print(f"ONNX Optimized:    {onnx_time*1000:.2f} ms/image ({1/onnx_time:.1f} FPS)")
    
    if onnx_time < 0.5:
        print("\n✅ REQUIREMENT MET: Inference is < 0.5 seconds.")
    else:
        print("\n⚠️ WARNING: Inference is slower than target.")
        
    print(f"🚀 Speedup Factor: {pt_time/onnx_time:.2f}x")

if __name__ == "__main__":
    # Install onnxruntime if missing
    try:
        import onnxruntime
    except ImportError:
        import os
        os.system('pip install onnx onnxruntime-gpu')
        
    export_to_onnx()