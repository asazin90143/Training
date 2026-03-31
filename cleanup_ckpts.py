import os
import glob

def main():
    ckpt_dir = r'models\separation\sepformer_dann'
    ckpts = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
    
    if not ckpts:
        print("No checkpoints found.")
        return

    # Sort checkpoints by epoch number
    ckpts.sort(key=lambda x: int(os.path.basename(x).replace('checkpoint_epoch_', '').replace('.pt', '')))
    
    # Keep the latest 3 checkpoints
    keep = ckpts[-3:]
    
    removed_size = 0
    count = 0
    
    print(f"Total checkpoints found: {len(ckpts)}")
    print(f"Keeping latest 3 checkpoints: {[os.path.basename(k) for k in keep]}")
    
    for c in ckpts:
        if c not in keep:
            try:
                size = os.path.getsize(c)
                os.remove(c)
                removed_size += size
                count += 1
            except Exception as e:
                print(f"Failed to delete {c}: {e}")
                
    print(f"\nCleanup complete: Deleted {count} older checkpoints.")
    print(f"Recovered Disk Space: {removed_size / (1024**3):.2f} GB.")

if __name__ == '__main__':
    main()
