"""
Make sure tha the sky labels are empty and they are there. Check how many sky images vs the total labels.
"""

from pathlib import Path

def fix_missing_labels(img_dir: str, lbl_dir: str):
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    fixed = 0
    for img_path in img_dir.glob("sky_*.*"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():  # just check existence, not size
            lbl_path.write_text("")
            fixed += 1
    print(f"Fixed {fixed} missing label files in {lbl_dir}")

fix_missing_labels(
    "data/data3channels/train/images",
    "data/data3channels/train/labels"
)
fix_missing_labels(
    "data/data3channels/valid/images",
    "data/data3channels/valid/labels"
)

# Verify
for split in ["train", "valid"]:
    lbl_dir = Path(f"data/data3channels/{split}/labels")
    all_labels = list(lbl_dir.glob("*.txt"))
    empty = [f for f in all_labels if f.stat().st_size == 0]
    sky_imgs = list(Path(f"data/data3channels/{split}/images").glob("sky_*.*"))
    print(f"\n{split}:")
    print(f"  sky images : {len(sky_imgs)}")
    print(f"  empty labels: {len(empty)}")
    print(f"  total labels: {len(all_labels)}")