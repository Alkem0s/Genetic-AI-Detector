import os
import hashlib
from pathlib import Path
import concurrent.futures
from collections import defaultdict
import time

def get_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return f"Error: {e}"

def check_overlaps(root_dir):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: {root_dir} does not exist.")
        return

    print(f"Scanning for 'real/nature' images in {root_dir}...")
    
    # Map to store: hash -> list of (generator, split, filename)
    hash_map = defaultdict(list)
    
    # Find all 'real' or 'nature' folders
    real_folders = []
    for gen_dir in root_path.iterdir():
        if not gen_dir.is_dir():
            continue
            
        for split in ["train", "val"]:
            for cls_folder in ["nature", "real", "human"]:
                target = gen_dir / split / cls_folder
                if target.exists():
                    real_folders.append((gen_dir.name, split, target))

    if not real_folders:
        print("No 'real' image folders found. Checked for 'nature', 'real', and 'human' subfolders.")
        return

    print(f"Found {len(real_folders)} real image folders. Hashing files...")
    
    total_files = 0
    all_tasks = []
    
    # Prepare hashing tasks
    for gen_name, split, folder in real_folders:
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        for f in files:
            all_tasks.append((f, gen_name, split))
            total_files += 1

    print(f"Total images to hash: {total_files}")
    
    # Hash in parallel
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        future_to_info = {executor.submit(get_file_hash, task[0]): task for task in all_tasks}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_info):
            file_path, gen_name, split = future_to_info[future]
            file_hash = future.result()
            
            if not file_hash.startswith("Error"):
                hash_map[file_hash].append({
                    "generator": gen_name,
                    "split": split,
                    "name": file_path.name
                })
            
            completed += 1
            if completed % 1000 == 0:
                print(f"  Progress: {completed}/{total_files}...", end='\r')

    duration = time.time() - start_time
    print(f"\nHash calculation complete in {duration:.2f}s")

    # Analyze overlaps
    overlap_count = 0
    generator_overlaps = defaultdict(lambda: defaultdict(int)) # gen1 -> gen2 -> count
    
    # Analyze by Hash
    print("\n--- Overlap Analysis (by MD5 Hash) ---")
    hash_overlaps = 0
    for occurrences in hash_map.values():
        if len(occurrences) > 1:
            generators = {occ['generator'] for occ in occurrences}
            if len(generators) > 1:
                hash_overlaps += 1
    print(f"Binary identical images (same hash) across generators: {hash_overlaps}")

    # Analyze by Filename
    print("\n--- Overlap Analysis (by Filename) ---")
    name_map = defaultdict(list)
    for file_hash, occurrences in hash_map.items():
        for occ in occurrences:
            name_map[occ['name']].append(occ)

    filename_overlaps = 0
    for name, occurrences in name_map.items():
        if len(occurrences) > 1:
            generators = {occ['generator'] for occ in occurrences}
            if len(generators) > 1:
                filename_overlaps += 1
                gen_list = sorted(list(generators))
                for i in range(len(gen_list)):
                    for j in range(i + 1, len(gen_list)):
                        generator_overlaps[gen_list[i]][gen_list[j]] += 1

    if filename_overlaps == 0:
        print("No overlapping filenames found across different generators.")
    else:
        print(f"Found {filename_overlaps} filenames that are shared between multiple generators.")
        print("This is a strong indicator of shared source images, even if the files themselves differ.")
        
        # Detail the overlaps
        print("\n--- Detailed Filename Overlaps ---")
        for name, occurrences in name_map.items():
            if len(occurrences) > 1:
                generators = {occ['generator'] for occ in occurrences}
                if len(generators) > 1:
                    print(f"  File: {name}")
                    for occ in occurrences:
                        print(f"    - {occ['generator']} ({occ['split']})")

    # Detail Hash Overlaps
    if hash_overlaps > 0:
        print("\n--- Detailed Hash Overlaps (Identical Files) ---")
        for file_hash, occurrences in hash_map.items():
            if len(occurrences) > 1:
                generators = {occ['generator'] for occ in occurrences}
                if len(generators) > 1:
                    print(f"  Hash: {file_hash}")
                    for occ in occurrences:
                        print(f"    - {occ['generator']} ({occ['split']}): {occ['name']}")

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    check_overlaps(target)
