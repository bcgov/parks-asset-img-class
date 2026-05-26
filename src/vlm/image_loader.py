import os
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent 

def load_asset_images_base64(asset_id, df, image_root):
    """
    Load all images for a given asset and return:
    [{"b64": ..., "mime": ...}, ...]
    """
    paths = (
        df[df["asset_id"] == asset_id]["image_path"]
        .dropna()
        .tolist()
    )

    images = []
    for p in paths:
        fixed = ROOT / p.replace("data/", f"{image_root}/", 1)
        if not os.path.exists(fixed):
            print(f"⚠️ Image not found: {fixed}")
            continue

        mime = "image/png" if str(fixed).lower().endswith(".png") else "image/jpeg"

        with open(fixed, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        images.append({"b64": b64, "mime": mime})

    return images