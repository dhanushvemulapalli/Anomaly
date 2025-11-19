# ela.py
from PIL import Image, ImageChops, ImageEnhance
import os

def compute_ela(image_path, quality=95, scale=30):
    """Compute the Error Level Analysis (ELA) image."""
    
    # Load original
    original = Image.open(image_path).convert('RGB')

    # Temporary recompressed image
    temp_path = "temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)

    # Load recompressed
    recompressed = Image.open(temp_path)

    # Pixel-wise difference
    ela_img = ImageChops.difference(original, recompressed)

    # Enhance for visibility
    enhancer = ImageEnhance.Brightness(ela_img)
    ela_img = enhancer.enhance(scale)

    return original, ela_img
