#!/usr/bin/env python3
"""
ELA CASIA Pipeline
- Batch-process images in INPUT_DIR
- Save: <basename>_ela.png, <basename>_predmask.png, <basename>_overlay.png
- If matching mask exists in MASK_DIR (same filename), compute metrics and write summary CSV

Usage:
    python ela_casia_pipeline.py --input_dir path/to/images --output_dir outputs --mask_dir path/to/masks

Dependencies:
    pip install pillow numpy pandas matplotlib
"""
import argparse
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
import io
import sys

# -------------------- Utils --------------------
def compute_ela_from_pil(img_pil, quality=95, scale=30):
    """Compute ELA image from a PIL RGB image. Returns (ela_pil, ela_np_float_0_1)."""
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img_pil.convert("RGB"), recompressed)
    diff_np = np.asarray(diff).astype(np.float32) / 255.0
    diff_np = np.clip(diff_np * scale, 0.0, 1.0)
    ela_uint8 = (diff_np * 255).astype(np.uint8)
    ela_pil = Image.fromarray(ela_uint8)
    return ela_pil, diff_np

def rgb_to_gray_np(img_np):
    """Convert RGB numpy (H,W,3) in 0..1 to grayscale 0..1."""
    return 0.2989 * img_np[...,0] + 0.5870 * img_np[...,1] + 0.1140 * img_np[...,2]

def otsu_threshold(gray_0_1):
    """Otsu's threshold for grayscale 0..1 image. Returns threshold in 0..1."""
    img = (gray_0_1 * 255).astype(np.uint8)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0,256))
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    current_max = 0.0
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
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = i
    return threshold / 255.0

def segment_ela(ela_np, method="otsu", manual_thresh=None, smooth_radius=1):
    """
    Segment ELA image (ela_np in 0..1, HxWx3 or HxW) to binary mask.
    Returns (mask_bool_HxW, used_threshold).
    Applies simple median smoothing (PIL) on grayscale before thresholding if smooth_radius>0.
    """
    if ela_np.ndim == 3:
        gray = rgb_to_gray_np(ela_np)
    else:
        gray = ela_np
    # Smooth using median filter (implemented by PIL)
    if smooth_radius and smooth_radius > 0:
        pil_gray = Image.fromarray((gray * 255).astype(np.uint8))
        pil_gray = pil_gray.filter(ImageFilter.MedianFilter(size=3*smooth_radius))
        gray = np.asarray(pil_gray).astype(np.float32) / 255.0
    if method == "otsu":
        t = otsu_threshold(gray)
    elif method == "manual":
        if manual_thresh is None:
            raise ValueError("manual_thresh must be set when method='manual'")
        t = float(manual_thresh)
    else:
        raise ValueError("Unknown segmentation method: " + str(method))
    mask = gray >= t
    return mask, t

def overlay_mask(original_pil, mask_bool, alpha=0.45, color=(255,0,0)):
    """Overlay binary mask (HxW bool) onto original PIL image and return PIL RGB."""
    orig = original_pil.convert("RGBA")
    w,h = orig.size
    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255).astype(np.uint8))
    if mask_img.size != orig.size:
        mask_img = mask_img.resize(orig.size)
    overlay = Image.new("RGBA", orig.size, (0,0,0,0))
    overlay_pixels = overlay.load()
    mask_px = mask_img.load()
    for y in range(h):
        for x in range(w):
            if mask_px[x,y] > 0:
                overlay_pixels[x,y] = (color[0], color[1], color[2], int(255*alpha))
    combined = Image.alpha_composite(orig, overlay)
    return combined.convert("RGB")

def evaluate_mask(pred_mask, gt_mask):
    """Pixel-wise metrics. pred_mask and gt_mask are boolean numpy arrays same shape."""
    pred = pred_mask.astype(np.uint8)
    gt = gt_mask.astype(np.uint8)
    tp = int(np.logical_and(pred==1, gt==1).sum())
    fp = int(np.logical_and(pred==1, gt==0).sum())
    fn = int(np.logical_and(pred==0, gt==1).sum())
    tn = int(np.logical_and(pred==0, gt==0).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    pixel_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {"IoU": iou, "Precision": precision, "Recall": recall, "F1": f1, "PixelAcc": pixel_acc,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}

def save_png(pil_img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(path)

# -------------------- Batch processing --------------------
def batch_process_dataset(input_dir, output_dir, mask_dir=None,
                          quality=95, scale=30, method="otsu", manual_thresh=None,
                          smooth_radius=1, overlay_alpha=0.45, write_csv=True, verbose=True):
    """
    Processes all files in input_dir. If mask_dir provided and contains a file with same name, evaluate metrics.
    Returns a pandas DataFrame with metrics for images that had GT masks (may be empty).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(mask_dir) if mask_dir is not None else None

    metrics_list = []
    supported = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    files = []
    for pat in supported:
        files.extend(sorted(input_dir.glob(pat)))
    if verbose:
        print(f"[+] Found {len(files)} files in {input_dir}")

    for idx, p in enumerate(files, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            if verbose:
                print(f"[!] Skipping {p.name}: {e}")
            continue

        ela_pil, ela_np = compute_ela_from_pil(img, quality=quality, scale=scale)
        base = p.stem
        ela_path = output_dir / f"{base}_ela.png"
        save_png(ela_pil, ela_path)

        mask_pred, used_t = segment_ela(ela_np, method=method, manual_thresh=manual_thresh, smooth_radius=smooth_radius)
        pred_mask_pil = Image.fromarray((mask_pred.astype(np.uint8) * 255).astype(np.uint8))
        pred_mask_path = output_dir / f"{base}_predmask.png"
        save_png(pred_mask_pil, pred_mask_path)

        overlay_pil = overlay_mask(img, mask_pred, alpha=overlay_alpha)
        overlay_path = output_dir / f"{base}_overlay.png"
        save_png(overlay_pil, overlay_path)

        if mask_dir is not None:
            gt_path = mask_dir / p.name
            # if not found by name, try stem with png
            if (not gt_path.exists()) and (mask_dir / (p.stem + ".png")).exists():
                gt_path = mask_dir / (p.stem + ".png")
            if gt_path.exists():
                try:
                    gt = Image.open(gt_path).convert("L")
                    gt_arr = np.asarray(gt) > 127
                    # Resize GT if different shape
                    if gt_arr.shape != mask_pred.shape:
                        gt_pil = Image.fromarray((gt_arr.astype(np.uint8)*255).astype(np.uint8))
                        gt_pil = gt_pil.resize(pred_mask_pil.size)
                        gt_arr = np.asarray(gt_pil) > 127
                    metrics = evaluate_mask(mask_pred, gt_arr)
                    metrics.update({"image": p.name, "threshold": float(used_t)})
                    metrics_list.append(metrics)
                except Exception as e:
                    if verbose:
                        print(f"[!] Could not evaluate {p.name} with GT {gt_path.name}: {e}")
        if verbose and (idx % 50 == 0 or idx == len(files)):
            print(f"[+] Processed {idx}/{len(files)} images")

    df = pd.DataFrame(metrics_list)
    if write_csv and not df.empty:
        df.to_csv(Path(output_dir)/"summary_metrics.csv", index=False)
        if verbose:
            print(f"[+] Wrote summary metrics to {Path(output_dir)/'summary_metrics.csv'}")
    elif write_csv:
        if verbose:
            print("[+] No GT masks found or no metrics to write.")
    return df

# -------------------- CLI --------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="ELA batch pipeline for CASIA")
    p.add_argument("--input_dir", required=True, help="Path to input images")
    p.add_argument("--output_dir", required=True, help="Path to save outputs")
    p.add_argument("--mask_dir", default=None, help="(optional) Path to ground-truth masks (matching filenames)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality for recompression (default:95)")
    p.add_argument("--scale", type=int, default=30, help="ELA amplification scale (default:30)")
    p.add_argument("--method", choices=["otsu","manual"], default="otsu", help="Segmentation method")
    p.add_argument("--manual_thresh", type=float, default=None, help="Manual threshold (0..1) if method=manual")
    p.add_argument("--smooth_radius", type=int, default=1, help="Median filter radius for smoothing (default:1)")
    p.add_argument("--overlay_alpha", type=float, default=0.45, help="Overlay alpha for visualization (default:0.45)")
    p.add_argument("--no_csv", dest="write_csv", action="store_false", help="Do not write summary CSV")
    p.add_argument("--quiet", action="store_true", help="Quiet mode")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    df = batch_process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        quality=args.quality,
        scale=args.scale,
        method=args.method,
        manual_thresh=args.manual_thresh,
        smooth_radius=args.smooth_radius,
        overlay_alpha=args.overlay_alpha,
        write_csv=args.write_csv,
        verbose=(not args.quiet)
    )
    if not df.empty:
        print(df.describe(include='all'))
    print("[+] Done.")
