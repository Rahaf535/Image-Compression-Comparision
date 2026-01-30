from pathlib import Path
from PIL import Image
import os

data_dir = Path("data")
out_dir  = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

imgs = list(data_dir.glob("*"))
if not imgs:
    raise SystemExit("Put at least one image inside the 'data' folder!")

src = Image.open(imgs[0]).convert("RGB")

jpeg = out_dir / "out.jpg"
webp_lossy = out_dir / "out_lossy.webp"
webp_lossless = out_dir / "out_lossless.webp"
png = out_dir / "out.png"

src.save(jpeg, "JPEG", quality=75, optimize=True)
src.save(webp_lossy, "WEBP", quality=75, method=6)
src.save(webp_lossless, "WEBP", lossless=True, quality=100, method=6)
src.save(png, "PNG", optimize=True, compress_level=6)

kb = lambda p: os.path.getsize(p) / 1024
print(f"Input: {imgs[0].name}")
print(f"JPEG (q=75):       {kb(jpeg):.1f} KB")
print(f"WebP (lossy q=75): {kb(webp_lossy):.1f} KB")
print(f"WebP (lossless):   {kb(webp_lossless):.1f} KB")
print(f"PNG (lossless):    {kb(png):.1f} KB")
