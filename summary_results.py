import pandas as pd

df = pd.read_csv("outputs/benchmark_multi_quality.csv")

lossy_df = df[df["codec"].isin(["jpeg", "webp"])]

summary = (
    lossy_df
    .groupby("codec")
    .agg(
        avg_bpp=("bpp", "mean"),
        avg_psnr=("psnr_db", "mean"),
        avg_ssim=("ssim", "mean")
    )
    .reset_index()
)

print(summary)
summary.to_csv("outputs/summary_results.csv", index=False)
