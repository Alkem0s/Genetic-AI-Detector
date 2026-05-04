import random
import shutil
import json
from pathlib import Path
import concurrent.futures
import time


def subsample_generator(
    data_root: str,
    generator_name: str,
    dst_root: str,
    n_train: int = 20000,
    n_val: int = 5000,
    seed: int = 42,
):
    """
    Subsamples from an already-extracted GenImage generator directory.
    
    Expects:
        data_root/generator_name/train/fake/
        data_root/generator_name/train/nature/  (or real/)
        data_root/generator_name/val/fake/
        data_root/generator_name/val/nature/
    """
    import PIL.Image
    random.seed(seed)
    is_image = lambda p: p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}

    data_root = Path(data_root)
    dst_root = Path(dst_root)
    generator_dir = data_root / generator_name
    if not generator_dir.exists():
        print(f"\n[!] Generator directory '{generator_dir}' not found. Skipping.")
        return None
    
    log = {}
    from global_config import image_size as min_dim

    for split, n_total in [("train", n_train), ("val", n_val)]:
        target_each = n_total // 2
        split_valid_images = {}

        for cls, possible_folders in [("ai", ["ai", "fake"]), ("real", ["nature", "real", "human"])]:
            src_dir = None
            for folder in possible_folders:
                candidate = generator_dir / split / folder
                if candidate.exists():
                    src_dir = candidate
                    break

            if src_dir is None:
                print(f"  [!] Skipping {generator_name} {split}/{cls}: folder not found.")
                continue

            print(f"  [{generator_name}] {split}/{cls}: scanning and filtering (min {min_dim}px)...", flush=True)
            scan_start = time.time()
            valid_images = []
            
            import os
            try:
                with os.scandir(src_dir) as it:
                    for entry in it:
                        if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            p = Path(entry.path)
                            try:
                                # Fast header peek
                                with PIL.Image.open(p) as img:
                                    w, h = img.size
                                    if w >= min_dim and h >= min_dim:
                                        valid_images.append(p)
                            except:
                                continue # Skip corrupted
                        
                        if (len(valid_images) + 1) % 50000 == 0:
                            print(f"    ...found {len(valid_images)} valid images so far...", end='\r', flush=True)
            except Exception as e:
                print(f"Error scanning {src_dir}: {e}")
            
            # Fallback to rglob if empty (rare)
            if not valid_images:
                print(f"    No images in root, trying recursive scan...", flush=True)
                for p in src_dir.rglob("*"):
                    if is_image(p):
                        try:
                            with PIL.Image.open(p) as img:
                                if all(dim >= min_dim for dim in img.size):
                                    valid_images.append(p)
                        except: continue

            print(f"  [{generator_name}] {split}/{cls}: {len(valid_images)} valid images found in {time.time()-scan_start:.1f}s")
            split_valid_images[cls] = valid_images

        # Balanced sampling logic
        if len(split_valid_images) < 2:
            continue
            
        available_ai = len(split_valid_images.get("ai", []))
        available_real = len(split_valid_images.get("real", []))
        
        # The bottleneck is the smaller of the two, capped by target_each
        actual_n = min(target_each, available_ai, available_real)
        print(f"  [{generator_name}] {split}: Sampling {actual_n} images from each class for 50/50 balance.")

        for cls in ["ai", "real"]:
            selected = random.sample(split_valid_images[cls], actual_n)
            dst_dir = dst_root / generator_name / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            print(f"    Copying {cls}...")
            start_time = time.time()
            def copy_one(img_path):
                try:
                    shutil.copy(img_path, dst_dir / img_path.name)
                except: return 1
                return 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                futures = [executor.submit(copy_one, img) for img in selected]
                for i, f in enumerate(concurrent.futures.as_completed(futures)):
                    if (i + 1) % 1000 == 0 or (i + 1) == actual_n:
                        print(f"      Progress: {i+1}/{actual_n}", end='\r', flush=True)
            
            print(f"\n    {cls} done in {time.time()-start_time:.1f}s")
            log[f"{split}_{cls}"] = actual_n

    print(f"\n[{generator_name}] Done.")
    return log


if __name__ == "__main__":
    for generator in ["ADM", "BigGAN", "glide", "Midjourney", "sdv4", "sdv5", "vqdm", "wukong"]:
        subsample_generator(
            data_root="data",
            generator_name=generator,
            dst_root="dataset_sampled",
            n_train=20000,
            n_val=5000,
            seed=42,
        )