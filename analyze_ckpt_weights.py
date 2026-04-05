import os
import glob
import torch

def analyze_weights():
    ckpt_dir = r'models\separation\sepformer_dann'
    ckpts = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
    
    if not ckpts:
        print("No checkpoints found.")
        return

    # Sort checkpoints by epoch number
    ckpts.sort(key=lambda x: int(os.path.basename(x).replace('checkpoint_epoch_', '').replace('.pt', '')))
    
    latest_ckpt = ckpts[-1]
    print(f"Analyzing Latest Checkpoint: {latest_ckpt}")
    
    try:
        state = torch.load(latest_ckpt, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
        
        with open("analysis_output.txt", "w", encoding="utf-8") as f:
            f.write(f"Analyzing Latest Checkpoint: {latest_ckpt}\n")
            f.write(f"Epoch Check: {state.get('epoch')}\n\n")
            
            # 1. Analyze Domain Classifier (Adversarial Check)
            d_state = state.get("domain_state", {})
            if d_state:
                f.write("[Domain Classifier Weights - DANN Health]\n")
                for k, v in d_state.items():
                    if "weight" in k:
                        f.write(f" - Layer: {k:30} | Mean: {v.mean().item():.5f} | Std: {v.std().item():.5f} | Range: [{v.min().item():.3f}, {v.max().item():.3f}]\n")
            else:
                f.write("Domain Classifier state not found.\n")

            # 2. Analyze SepFormer (Encoder & Masknet Health)
            s_state = state.get("sep_state", {})
            if s_state:
                f.write("\n[SepFormer Network Weights - Structural Health (Sampling)]\n")
                count = 0
                for k, v in s_state.items():
                    if "weight" in k and ('encoder' in k or 'att' in k or 'norm' in k):
                        if count < 10:
                            f.write(f" - Layer: {k:50} | Mean: {v.mean().item():.5f} | Std: {v.std().item():.5f} | Range: [{v.min().item():.3f}, {v.max().item():.3f}]\n")
                            count += 1
            else:
                f.write("\n[SepFormer Weights]: Not found in state dict.\n")
        
        print("Analysis complete. Saved to analysis_output.txt")

if __name__ == "__main__":
    analyze_weights()
