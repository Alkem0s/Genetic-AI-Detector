import os
from pathlib import Path
import PIL.Image
import global_config as config

def clean_dataset():
    base_dir = Path(config.dataset_sampled_dir)
    image_size = config.image_size
    deleted_count = 0
    total_checked = 0
    
    print(f"Cleaning dataset in {base_dir} (minimum size {image_size}x{image_size})")
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return
        
    for gen_path in base_dir.iterdir():
        if not gen_path.is_dir():
            continue
            
        for split in ["train", "val"]:
            split_path = gen_path / split
            if not split_path.exists():
                continue
                
            for cls_name in ["ai", "real"]:
                cls_path = split_path / cls_name
                if not cls_path.exists():
                    continue
                    
                all_imgs = [f for f in cls_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
                
                for img_file in all_imgs:
                    total_checked += 1
                    if total_checked % 5000 == 0:
                        print(f"Checked {total_checked} images...")
                    
                    is_valid = False
                    try:
                        with PIL.Image.open(img_file) as im:
                            w, h = im.size
                            if w >= image_size and h >= image_size:
                                is_valid = True
                    except Exception as e:
                        print(f"Error reading {img_file}: {e}")
                    
                    if not is_valid:
                        try:
                            os.remove(img_file)
                            deleted_count += 1
                        except Exception as e:
                            print(f"Failed to delete {img_file}: {e}")

    print(f"Done! Checked {total_checked} images and deleted {deleted_count} unsuitable/corrupted images.")

if __name__ == '__main__':
    clean_dataset()
