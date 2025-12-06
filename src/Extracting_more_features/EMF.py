import argparse
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
import io
import sys
import time
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy as scipy_entropy

# ------------------------------------------------------------
# ELA Computation
# ------------------------------------------------------------
def compute_ela(pil_img, quality=95, scale=15):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(pil_img, recompressed)

    ela_np = np.asarray(diff).astype(np.float32) / 255.0
    ela_np = np.clip(ela_np * scale, 0.0, 1.0)

    ela_uint8 = (ela_np * 255).astype(np.uint8)
    ela_pil = Image.fromarray(ela_uint8)

    return ela_pil, ela_np


# ------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------
def rgb_to_gray(arr):
    return 0.2989*arr[...,0] + 0.5870*arr[...,1] + 0.1140*arr[...,2]

def otsu(gray):
    img = (gray * 255).astype(np.uint8)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0,256))

    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = 0
    threshold = 0

    for i in range(256):
        w_b += hist[i]
        if w_b == 0: 
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f)**2

        if var_between > max_var:
            max_var = var_between
            threshold = i

    return threshold / 255.0


def segment(ela_np):
    gray = rgb_to_gray(ela_np)
    pil_gray = Image.fromarray((gray*255).astype(np.uint8))
    pil_gray = pil_gray.filter(ImageFilter.MedianFilter(size=3))
    gray = np.asarray(pil_gray).astype(np.float32) / 255.0
    t = otsu(gray)
    mask = gray >= t
    return mask, t

def extract_features(ela_np, mask):
    """
    Compute statistical + texture feature set from ELA output.
    Only masked (suspicious) regions are analyzed.
    """
    # Convert to grayscale for texture analysis
    gray = rgb_to_gray(ela_np)
    region = gray[mask] if mask.sum() > 0 else gray
    
    # Basic statistics
    mean_intensity = float(region.mean())
    std_intensity = float(region.std())

    # Histogram-based features
    hist, _ = np.histogram(region, bins=256, range=(0,1), density=True)
    peak_indices = hist.argsort()[-3:][::-1]  # top 3 histogram peaks

    # Texture (GLCM Haralick features)
    img_8bit = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(img_8bit, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = float(graycoprops(glcm, 'contrast')[0,0])
    energy = float(graycoprops(glcm, 'energy')[0,0])

    # Entropy (information content)
    hist_safe = hist + 1e-12
    ent = float(scipy_entropy(hist_safe))

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "contrast": contrast,
        "entropy": ent,
        "energy": energy,
        "hist_peak_1": int(peak_indices[0]),
        "hist_peak_2": int(peak_indices[1]),
        "hist_peak_3": int(peak_indices[2])
    }

# ------------------------------------------------------------
# Overlay
# ------------------------------------------------------------
def overlay_mask(original, mask, alpha=0.4):
    orig = original.convert("RGBA")
    overlay = Image.new("RGBA", orig.size, (0,0,0,0))
    w,h = orig.size

    mask_img = Image.fromarray((mask.astype(np.uint8)*255).astype(np.uint8))
    mask_px = mask_img.load()
    over_px = overlay.load()

    for y in range(h):
        for x in range(w):
            if mask_px[x,y] > 0:
                over_px[x,y] = (255, 0, 0, int(alpha*255))

    return Image.alpha_composite(orig, overlay).convert("RGB")


# ------------------------------------------------------------
# Batch Processor for TP and AU
# ------------------------------------------------------------
def process_folder(folder_path, out_root, label):
    folder = Path(folder_path)
    out_dir = Path(out_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))

    for p in files:
        img = Image.open(p).convert("RGB")
        ela_img, ela_np = compute_ela(img)

        base = p.stem
        ela_img.save(out_dir / f"{base}_ela.png")

        mask, t = segment(ela_np)
        mask_img = Image.fromarray((mask.astype(np.uint8)*255).astype(np.uint8))
        mask_img.save(out_dir / f"{base}_mask.png")

        overlay = overlay_mask(img, mask)
        overlay.save(out_dir / f"{base}_overlay.png")

        # feature score = mean ELA intensity
        # feature score = mean ELA intensity
        anomaly_score = float(ela_np.mean())

        # ---- NEW FEATURE EXTRACTION ----
        feat = extract_features(ela_np, mask)

        rows.append({
            "image": p.name,
            "class": label,
            "threshold": t,
            "anomaly_score": anomaly_score,
            **feat   # merges new feature fields
        })


    return rows


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="data directory containing TP and AU")
    parser.add_argument("--output", required=True, help="output directory")

    
    args = parser.parse_args()

    root = Path(args.data_dir)

    tp_dir = root / "TP"
    au_dir = root / "AU"

    rows = []

    if tp_dir.exists():
        print("[+] Processing TP...")
        rows.extend(process_folder(tp_dir, args.output, "TP"))

    if au_dir.exists():
        print("[+] Processing AU...")
        rows.extend(process_folder(au_dir, args.output, "AU"))

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.output)/"summary.csv", index=False)
    print("[+] Done. Saved summary.csv")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"[+] Total time taken: {end - start:.2f} seconds")
