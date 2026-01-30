from pathlib import Path
from PIL import Image
import numpy as np
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BASE = Path(__file__).resolve().parent
data_dir = BASE / "data"
out_dir  = BASE / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

imgs = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
if not imgs:
    raise SystemExit("Put at least one image inside the 'data' folder!")
ref_path = imgs[0]  
ref = Image.open(ref_path).convert("RGB")
ref_np = np.array(ref)
H, W = ref_np.shape[:2]

candidates = [
    ("JPEG q=75",        out_dir / "out.jpg"),
    ("WebP lossy q=75",  out_dir / "out_lossy.webp"),
    ("WebP lossless",    out_dir / "out_lossless.webp"),
    ("PNG lossless",     out_dir / "out.png"),
]

def bits_per_pixel(size_bytes: int, w: int, h: int) -> float:
    return (size_bytes * 8.0) / (w * h)

rows = []
for label, path in candidates:
    if not path.exists():
        rows.append({
            "format": label,
            "exists": False,
            "size_kb": None,
            "bpp": None,
            "psnr_db": None,
            "ssim": None,
        })
        continue

    img = Image.open(path).convert("RGB")
    img_np = np.array(img)

    psnr_val = psnr(ref_np, img_np, data_range=255)
    ssim_val = ssim(ref_np, img_np, data_range=255, channel_axis=2)

    size_bytes = os.path.getsize(path)
    row = {
        "format": label,
        "exists": True,
        "size_kb": size_bytes / 1024.0,
        "bpp": bits_per_pixel(size_bytes, W, H),
        "psnr_db": psnr_val,
        "ssim": ssim_val,
    }
    rows.append(row)

df = pd.DataFrame(rows, columns=["format", "exists", "size_kb", "bpp", "psnr_db", "ssim"])

print(f"Reference: {ref_path.name}  ({W}x{H})")
print(df.to_string(index=False, formatters={
    "size_kb": lambda x: "—" if x is None else f"{x:8.1f}",
    "bpp":     lambda x: "—" if x is None else f"{x:6.3f}",
    "psnr_db": lambda x: "—" if x is None else f"{x:7.2f}",
    "ssim":    lambda x: "—" if x is None else f"{x:6.4f}",
}))

csv_path = out_dir / "metrics_current_outputs.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved metrics to: {csv_path}")
