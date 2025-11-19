# visualize.py
import matplotlib.pyplot as plt

def show_side_by_side(original, ela_img, title="ELA Detection"):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("ELA Output (Anomaly Map)")
    plt.imshow(ela_img, cmap="hot")
    plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
