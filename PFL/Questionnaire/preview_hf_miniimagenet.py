# Preview selected Mini-ImageNet classes.
import os
from collections import defaultdict
from datasets import load_dataset
from PIL import Image, ImageOps, ImageDraw


# Target WNIDs and readable class names.
TARGET_WNIDS = {
    "n02099601": "golden_retriever",  
    "n02110063": "malamute",          
    "n02110341": "dalmatian",         
    "n02120079": "arctic_fox",        
    "n02129165": "lion",              
}

OUT_DIR = "/home/zyq/PFLlib/system/miniimagenet_preview"
NUM_PER_CLASS = 8
CACHE_DIR = "/home/zyq/PFLlib/system/data/hf_cache"


def make_grid(saved, out_path, thumb_size=(160, 160), padding=12, header_h=35):
    # Create a visual grid for saved samples.
    rows = len(saved)
    cols = max(len(v) for v in saved.values())

    w = padding + cols * (thumb_size[0] + padding)
    h = padding + rows * (thumb_size[1] + header_h + padding)

    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    y = padding
    for wnid, paths in saved.items():
        title = f"{wnid}  {TARGET_WNIDS.get(wnid, '')}"
        draw.text((padding, y + 8), title, fill=(0, 0, 0))

        x = padding
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = ImageOps.contain(img, thumb_size)

            bg = Image.new("RGB", thumb_size, (240, 240, 240))
            bg.paste(img, ((thumb_size[0] - img.width) // 2, (thumb_size[1] - img.height) // 2))

            canvas.paste(bg, (x, y + header_h))
            x += thumb_size[0] + padding

        y += thumb_size[1] + header_h + padding

    canvas.save(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load Mini-ImageNet in streaming mode.
    ds = load_dataset(
        "timm/mini-imagenet",
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR
    )

    label_feature = ds.features["label"]
    label_names = getattr(label_feature, "names", None)

    saved = defaultdict(list)

    for item in ds:
        label = item["label"]

        if label_names is not None:
            wnid = label_names[label]
        else:
            wnid = str(label)

        if wnid not in TARGET_WNIDS:
            continue

        if len(saved[wnid]) >= NUM_PER_CLASS:
            continue

        class_name = TARGET_WNIDS[wnid]
        save_dir = os.path.join(OUT_DIR, f"{wnid}_{class_name}")
        os.makedirs(save_dir, exist_ok=True)

        img = item["image"].convert("RGB")
        save_path = os.path.join(save_dir, f"{len(saved[wnid]) + 1:02d}.jpg")
        img.save(save_path)

        saved[wnid].append(save_path)
        print(f"[Saved] {wnid} {class_name}: {save_path}")

        if all(len(saved[w]) >= NUM_PER_CLASS for w in TARGET_WNIDS):
            break

    print("\nSaved samples:")
    for wnid, class_name in TARGET_WNIDS.items():
        print(f"{wnid:10s} {class_name:18s}: {len(saved[wnid])} images")

    if saved:
        grid_path = os.path.join(OUT_DIR, "selected_classes_grid.jpg")
        make_grid(saved, grid_path)
        print("\nOverview grid:", grid_path)
    else:
        print("\nNo images were saved. This HuggingFace version may not contain the specified WNIDs.")


if __name__ == "__main__":
    main()