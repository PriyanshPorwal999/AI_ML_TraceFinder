import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
from collections import defaultdict

# === CONFIG ===
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Official")

def get_image_data(base_dir=BASE_DIR):
    """
    Recursively scan dataset folders and collect image counts per class.
    """
    class_counts = defaultdict(int)
    sample_images = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp"))]
        if image_files:
            class_name = os.path.relpath(root, base_dir)
            class_counts[class_name] += len(image_files)
            random.shuffle(image_files)
            sample_paths = [os.path.join(root, f) for f in image_files[:3]]  # take 3 samples per class
            sample_images[class_name].extend(sample_paths)

    return class_counts, sample_images


def plot_class_distribution(class_counts):
    """
    Plots the number of images per class as a bar chart.
    """
    df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Image Count"])
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Class", y="Image Count", data=df, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Class Distribution — Number of Images per Scanner Type", fontsize=14)
    plt.tight_layout()
    plt.show()


def show_sample_images(sample_images):
    """
    Display a few random sample images from each class.
    """
    for class_name, paths in sample_images.items():
        print(f"\n📸 Sample images from: {class_name}")
        fig, axes = plt.subplots(1, len(paths), figsize=(15, 4))
        if len(paths) == 1:
            axes = [axes]
        for ax, img_path in zip(axes, paths):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(os.path.basename(img_path))
                ax.axis("off")
        plt.suptitle(class_name, fontsize=12)
        plt.show()
        

def main():
    print(f"📂 Scanning dataset in: {BASE_DIR}")
    class_counts, sample_images = get_image_data(BASE_DIR)
    
    print("\n✅ Total Classes Found:", len(class_counts))
    for cls, count in class_counts.items():
        print(f"{cls}: {count} images")

    # Plot distribution
    plot_class_distribution(class_counts)

    # Show sample images
    show_sample_images(sample_images)


if __name__ == "__main__":
    main()

def get_dataset_summary(DATA_DIR):
    import os, cv2, random
    import pandas as pd
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")

    data_info = []
    extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]

    for root, _, files in os.walk(DATA_DIR):
        img_files = [f for f in files if os.path.splitext(f)[1].lower() in extensions]
        if img_files:
            class_name = os.path.relpath(root, DATA_DIR)
            for f in img_files:
                path = os.path.join(root, f)
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                        fmt = img.format
                        data_info.append({
                            "Class": class_name,
                            "Path": path,
                            "Width": width,
                            "Height": height,
                            "Format": fmt
                        })
                except Exception:
                    pass

    if not data_info:
        return None, None, None

    df = pd.DataFrame(data_info)
    class_counts = df["Class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Image Count"]

    avg_width = df["Width"].mean()
    avg_height = df["Height"].mean()
    most_common_fmt = df["Format"].mode()[0]

    return df, class_counts, {
        "total_images": len(df),
        "total_classes": df["Class"].nunique(),
        "avg_resolution": f"{avg_width:.0f} × {avg_height:.0f}",
        "common_format": most_common_fmt,
    }

