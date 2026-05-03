import random
import zipfile
from pathlib import Path
from collections import defaultdict
import json
import os

def is_image(name, exts=(".jpg", ".jpeg", ".png", ".webp")):
    return name.lower().endswith(exts)


def extract_balanced_from_zip(
    zip_path,
    dst_root,
    x_train,
    y_val,
    seed=42,
    log_dict=None
):
    random.seed(seed)
    zip_path = Path(zip_path)
    dst_root = Path(dst_root)

    arch_name = zip_path.stem

    stats = {
        "train_ai": 0,
        "train_nature": 0,
        "val_ai": 0,
        "val_nature": 0
    }

    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = [f for f in z.namelist() if is_image(f)]

        def split(files, split_tag):
            ai = [f for f in files if f"/{split_tag}/ai/" in f]
            nat = [f for f in files if f"/{split_tag}/nature/" in f]

            random.shuffle(ai)
            random.shuffle(nat)

            return ai, nat

        def extract(files, split, cls, n):
            selected = files[:n]
            out_dir = dst_root / arch_name / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            for f in selected:
                dst_path = out_dir / Path(f).name

                # avoid overwrite
                if dst_path.exists():
                    base = dst_path.stem
                    ext = dst_path.suffix
                    i = 1
                    while dst_path.exists():
                        dst_path = out_dir / f"{base}_{i}{ext}"
                        i += 1

                with z.open(f) as src, open(dst_path, "wb") as out:
                    out.write(src.read())

            return len(selected)

        # ---- TRAIN ----
        train_files = [f for f in all_files if "/train/" in f]
        train_ai, train_nat = split(train_files, "train")

        half_train = x_train // 2

        stats["train_ai"] = extract(train_ai, "train", "ai", half_train)
        stats["train_nature"] = extract(train_nat, "train", "nature", x_train - half_train)

        # ---- VAL ----
        val_files = [f for f in all_files if "/val/" in f]
        val_ai, val_nat = split(val_files, "val")

        half_val = y_val // 2

        stats["val_ai"] = extract(val_ai, "val", "ai", half_val)
        stats["val_nature"] = extract(val_nat, "val", "nature", y_val - half_val)

        print(f"[{arch_name}] {stats}")

        if log_dict is not None:
            log_dict[arch_name] = stats


def process_all_zips(src_folder, dst_root, x_train=20000, y_val=5000):
    src_folder = Path(src_folder)
    dst_root = Path(dst_root)

    zip_files = list(src_folder.rglob("*.zip"))

    all_logs = {}

    for zf in zip_files:
        extract_balanced_from_zip(
            zip_path=zf,
            dst_root=dst_root,
            x_train=x_train,
            y_val=y_val,
            log_dict=all_logs
        )

    # save logs for paper
    log_path = dst_root / "sampling_log.json"
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)

    print(f"\nSaved log to: {log_path}")


if __name__ == "__main__":
    process_all_zips(
        src_folder="PATH_TO_ZIP_ROOT",
        dst_root="OUTPUT_DATASET",
        x_train=20000,
        y_val=5000
    )