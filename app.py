import streamlit as st
from PIL import Image
import numpy as np
import os
import tempfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


st.set_page_config(
    page_title="Image Compression Comparison",
    layout="wide"
)


TMP_DIR = Path(tempfile.gettempdir()) / "image_compression_ui"
TMP_DIR.mkdir(exist_ok=True)


st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

quality = st.sidebar.slider(
    "Lossy Quality",
    10, 95, 75
)

show_diff = st.sidebar.checkbox("Show difference images", value=True)
show_rd = st.sidebar.checkbox("Show RD point", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("JPEG & WebP are lossy • PNG & WebP-lossless are lossless")


st.title("CompVis")
st.markdown(
    "**This interface allows you to upload an image and compare multiple image "
    "compression algorithms in terms of file size, visual distortion, and objective "
    "quality metrics.**"
)

def bits_per_pixel(size_bytes, w, h):
    return (size_bytes * 8.0) / (w * h)


if uploaded_file:
    ref_img = Image.open(uploaded_file).convert("RGB")
    ref_np = np.array(ref_img)
    W, H = ref_img.size

    st.subheader("Original Image")
    st.image(ref_img, width=350)

    results = []

    def process(codec, save_kwargs, ext, lossless=False):
        out_path = TMP_DIR / f"{codec}.{ext}"
        ref_img.save(out_path, **save_kwargs)
        img = Image.open(out_path).convert("RGB")
        img_np = np.array(img)
        size = os.path.getsize(out_path)

        return {
            "Codec": codec,
            "Size (KB)": size / 1024,
            "BPP": bits_per_pixel(size, W, H),
            "PSNR": float("inf") if lossless else psnr(ref_np, img_np, data_range=255),
            "SSIM": 1.0 if lossless else ssim(ref_np, img_np, data_range=255, channel_axis=2),
            "Image": img,
            "Path": out_path,
            "Diff": np.abs(ref_np.astype(int) - img_np.astype(int)).astype(np.uint8)
        }

    results.append(process(
        "JPEG",
        dict(format="JPEG", quality=quality, optimize=True),
        "jpg"
    ))

    results.append(process(
        "WebP_lossy",
        dict(format="WEBP", quality=quality, method=6),
        "webp"
    ))

    results.append(process(
        "WebP_lossless",
        dict(format="WEBP", lossless=True),
        "webp",
        lossless=True
    ))

    results.append(process(
        "PNG",
        dict(format="PNG", optimize=True),
        "png",
        lossless=True
    ))

    st.subheader("Compressed Results")

    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        with col:
            st.markdown(f"### {r['Codec']}")
            st.image(r["Image"])

            st.metric("Size (KB)", f"{r['Size (KB)']:.1f}")
            st.metric("BPP", f"{r['BPP']:.3f}")
            st.metric("PSNR (dB)", f"{r['PSNR']:.2f}")
            st.metric("SSIM", f"{r['SSIM']:.4f}")

            st.download_button(
                "Download",
                data=open(r["Path"], "rb"),
                file_name=r["Path"].name
            )

            if show_diff:
                st.caption("Difference image")
                st.image(r["Diff"], clamp=True)


    st.subheader("Summary Table")

    df = pd.DataFrame([{
        "Codec": r["Codec"],
        "Size (KB)": r["Size (KB)"],
        "BPP": r["BPP"],
        "PSNR (dB)": r["PSNR"],
        "SSIM": r["SSIM"]
    } for r in results])

    st.dataframe(df, use_container_width=True)


    if show_rd:
        st.subheader("Rate–Distortion Point (Uploaded Image)")

        fig = plt.figure()
        for r in results:
            plt.scatter(r["BPP"], r["PSNR"], label=r["Codec"])

        plt.xlabel("Bits per Pixel (bpp)")
        plt.ylabel("PSNR (dB)")
        plt.legend()
        plt.grid(True)

        st.pyplot(fig)

else:
    st.info("Upload an image using the sidebar to begin.")
