import os
import glob
import torch
import json

ckpt_dir = r'models\separation\sepformer_dann'
ckpts = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
ckpts.sort(key=lambda x: int(os.path.basename(x).replace('checkpoint_epoch_', '').replace('.pt', '')))
latest_ckpt = ckpts[-1]
state = torch.load(latest_ckpt, map_location="cpu")

out = {"domain": {}, "sepformer": {}}

d_state = state.get("domain_state", {})
for k in ["0.weight", "3.weight", "6.weight"]:
    if k in d_state:
        v = d_state[k]
        out["domain"][k] = {
            "mean": round(float(v.mean().item()), 4),
            "std": round(float(v.std().item()), 4),
            "min": round(float(v.min().item()), 4),
            "max": round(float(v.max().item()), 4)
        }

s_state = state.get("sep_state", {})
for k in list(s_state.keys())[:3]:
    v = s_state[k]
    out["sepformer"][k] = {
        "mean": round(float(v.mean().item()), 4),
        "std": round(float(v.std().item()), 4),
        "min": round(float(v.min().item()), 4),
        "max": round(float(v.max().item()), 4)
    }

with open("health_check_payload.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("JSON saved successfully.")
