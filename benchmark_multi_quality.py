from pathlib import Path
from PIL import Image
import numpy as np
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
data_dir = BASE / "data"
out_dir  = BASE / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)


QUALS = [10, 30, 50, 70, 90]


imgs = sorted([p for p in data_dir.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
if not imgs:
    raise SystemExit("Put at least one image inside the 'data' folder!")

def bits_per_pixel(size_bytes, w, h):
    return (size_bytes * 8.0) / (w * h)

rows = []

for img_path in imgs:
    name = img_path.stem
    ref = Image.open(img_path).convert("RGB")
    ref_np = np.array(ref)
    H, W = ref_np.shape[:2]

    
    for q in QUALS:
        
        jpeg_dir = out_dir / "jpeg" / f"q{q}"
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        jpeg_out = jpeg_dir / f"{name}.jpg"
        ref.save(jpeg_out, format="JPEG", quality=q, optimize=True)

        cand = Image.open(jpeg_out).convert("RGB")
        cand_np = np.array(cand)
        size_bytes = os.path.getsize(jpeg_out)

        rows.append({
            "image": img_path.name,
            "codec": "jpeg",
            "quality": q,
            "size_kb": size_bytes / 1024.0,
            "bpp": bits_per_pixel(size_bytes, W, H),
            "psnr_db": psnr(ref_np, cand_np, data_range=255),
            "ssim": ssim(ref_np, cand_np, data_range=255, channel_axis=2),
        })

        
        webp_dir = out_dir / "webp" / f"q{q}"
        webp_dir.mkdir(parents=True, exist_ok=True)
        webp_out = webp_dir / f"{name}.webp"
        ref.save(webp_out, format="WEBP", quality=q, method=6)

        cand = Image.open(webp_out).convert("RGB")
        cand_np = np.array(cand)
        size_bytes = os.path.getsize(webp_out)

        rows.append({
            "image": img_path.name,
            "codec": "webp",
            "quality": q,
            "size_kb": size_bytes / 1024.0,
            "bpp": bits_per_pixel(size_bytes, W, H),
            "psnr_db": psnr(ref_np, cand_np, data_range=255),
            "ssim": ssim(ref_np, cand_np, data_range=255, channel_axis=2),
        })

    
    
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png_out = png_dir / f"{name}.png"
    ref.save(png_out, format="PNG", optimize=True)

    size_bytes = os.path.getsize(png_out)
    rows.append({
        "image": img_path.name,
        "codec": "png",
        "quality": "lossless",
        "size_kb": size_bytes / 1024.0,
        "bpp": bits_per_pixel(size_bytes, W, H),
        "psnr_db": float("inf"),
        "ssim": 1.0,
    })

    
    webpll_dir = out_dir / "webp_lossless"
    webpll_dir.mkdir(parents=True, exist_ok=True)
    webpll_out = webpll_dir / f"{name}.webp"
    ref.save(webpll_out, format="WEBP", lossless=True, quality=100, method=6)

    size_bytes = os.path.getsize(webpll_out)
    rows.append({
        "image": img_path.name,
        "codec": "webp_lossless",
        "quality": "lossless",
        "size_kb": size_bytes / 1024.0,
        "bpp": bits_per_pixel(size_bytes, W, H),
        "psnr_db": float("inf"),
        "ssim": 1.0,
    })


df = pd.DataFrame(rows, columns=["image","codec","quality","size_kb","bpp","psnr_db","ssim"])
csv_path = out_dir / "benchmark_multi_quality.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")


for img_name, g in df.groupby("image"):
    
    fig1 = plt.figure()
    for codec in g["codec"].unique():
        gg = g[g["codec"] == codec].sort_values("bpp")
        plt.plot(gg["bpp"], gg["psnr_db"], marker="o", label=codec.upper())
    plt.xlabel("bpp (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Rate–Distortion (PSNR): {img_name}")
    plt.legend()
    fig1.savefig(out_dir / f"RD_PSNR_{Path(img_name).stem}.png", dpi=150)
    plt.close(fig1)

    
    fig2 = plt.figure()
    for codec in g["codec"].unique():
        gg = g[g["codec"] == codec].sort_values("bpp")
        plt.plot(gg["bpp"], gg["ssim"], marker="o", label=codec.upper())
    plt.xlabel("bpp (bits per pixel)")
    plt.ylabel("SSIM")
    plt.title(f"Rate–Distortion (SSIM): {img_name}")
    plt.legend()
    fig2.savefig(out_dir / f"RD_SSIM_{Path(img_name).stem}.png", dpi=150)
    plt.close(fig2)

print("Plots saved next to the CSV.")
