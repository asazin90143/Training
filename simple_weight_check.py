import os
import glob
import torch

ckpt_dir = r'models\separation\sepformer_dann'
ckpts = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
ckpts.sort(key=lambda x: int(os.path.basename(x).replace('checkpoint_epoch_', '').replace('.pt', '')))
latest_ckpt = ckpts[-1]
state = torch.load(latest_ckpt, map_location="cpu")

print("=== DOMAIN CLASSIFIER ===")
d_state = state.get("domain_state", {})
for k in ["0.weight", "3.weight", "6.weight"]:
    if k in d_state:
        v = d_state[k]
        print(f"{k} Mean: {v.mean().item():.4f} Std: {v.std().item():.4f}")

print("=== SEPFORMER ENCODER ===")
s_state = state.get("sep_state", {})
keys = list(s_state.keys())
for k in keys[:3]:
    v = s_state[k]
    print(f"{k} Range: [{v.min().item():.2f}, {v.max().item():.2f}]")
