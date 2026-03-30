import torch
import os

def analyze_weights():
    ckpt_path = r'models\separation\sepformer_dann\checkpoint_epoch_1461.pt'
    if not os.path.exists(ckpt_path):
        # Try to find the latest if 1461 doesn't exist yet
        import glob
        ckpts = glob.glob(r'models\separation\sepformer_dann\checkpoint_epoch_*.pt')
        if not ckpts:
            print("No checkpoints found.")
            return
        ckpt_path = max(ckpts, key=os.path.getmtime)
    
    print(f"Analyzing: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    print(f"Epoch: {state.get('epoch')}")
    
    # Analyze Domain Classifier
    d_state = state.get("domain_state", {})
    print("\n[Domain Classifier Weights]")
    for k, v in d_state.items():
        if "weight" in k:
            print(f" - {k:30} | Mean: {v.mean().item():.6f} | Std: {v.std().item():.6f} | Range: [{v.min().item():.4f}, {v.max().item():.4f}]")
    
    # Analyze SepFormer (encoder/feature extractor)
    s_state = state.get("sep_state", {})
    if s_state:
        print("\n[SepFormer Encoder Weights (Sampling)]")
        count = 0
        for k, v in s_state.items():
            if "weight" in k and count < 10:
                print(f" - {k:30} | Mean: {v.mean().item():.6f} | Std: {v.std().item():.6f} | Range: [{v.min().item():.4f}, {v.max().item():.4f}]")
                count += 1
    else:
        print("\n[SepFormer Weights]: Not found in state dict (Expected for unsupervised phase if not explicitly saved).")

if __name__ == "__main__":
    analyze_weights()
