import cv2
import os
import pandas as pd

# Path to the dataset folder (update this)
dataset_path = r"D:\OneDrive\Desktop\Project_Trace_Finder\dataset"

# Create a list to hold extracted features
data = []

# Loop over all files in dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".tif"):  # process only thumbnail images
            file_path = os.path.join(root, file)
            
            # Read image
            img = cv2.imread(file_path)
            
            if img is None:
                continue  # skip if image not read properly
            
            # Extract features
            image_name = file
            height, width, channels = img.shape
            resolution = width * height
            
            # Mean color values
            mean_color = cv2.mean(img)[:3]  # (B, G, R)
            
            # Device name mapping (example: from folder name)
            # You can refine this based on dataset structure
            device_name = os.path.basename(root)  
            
            # Append features to list
            data.append([
                image_name,
                device_name,
                resolution,
                width,
                height,
                channels,
                mean_color[2],  # R
                mean_color[1],  # G
                mean_color[0]   # B
            ])

# Convert to DataFrame
df = pd.DataFrame(data, columns=[
    "Image_Name", "Device_Name", "Resolution", 
    "Width", "Height", "Channels", "Mean_R", "Mean_G", "Mean_B"
])


# Folder for saving CSVs
output_folder = r"D:\OneDrive\Desktop\Project_Trace_Finder\FeatureOutputs"

# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save CSV inside FeatureOutputs
output_path = os.path.join(output_folder, "image_features.csv")
df.to_csv(output_path, index=False)

print(f"✅ Features extracted and saved to {output_path}")
