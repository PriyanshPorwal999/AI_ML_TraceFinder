# import cv2
# import os
# import pandas as pd
# import streamlit as st
# from PIL import Image

# # Streamlit page setup
# st.set_page_config(page_title="Image Feature Extractor", layout="wide")
# st.title("📸 Image Feature Extractor - Forgery Dataset Project")

# # User input: dataset path
# dataset_path = st.text_input("📂 Enter dataset folder path:", "")

# # Path to the dataset folder (update this)
# dataset_path = r"D:\OneDrive\Desktop\Project_Trace_Finder\dataset"

# # Create a list to hold extracted features
# data = []

# # Loop over all files in dataset
# for root, dirs, files in os.walk(dataset_path):
#     for file in files:
#         if file.lower().endswith(".tif"):  # process only thumbnail images
#             file_path = os.path.join(root, file)
            
#             # Read image
#             img = cv2.imread(file_path)
            
#             if img is None:
#                 continue  # skip if image not read properly
            
#             # Extract features
#             image_name = file
#             height, width, channels = img.shape
#             resolution = width * height
            
#             # Mean color values
#             mean_color = cv2.mean(img)[:3]  # (B, G, R)
            
#             # Device name mapping (example: from folder name)
#             # You can refine this based on dataset structure
#             device_name = os.path.basename(root)  
            
#             # Append features to list
#             data.append([
#                 image_name,
#                 device_name,
#                 resolution,
#                 width,
#                 height,
#                 channels,
#                 mean_color[2],  # R
#                 mean_color[1],  # G
#                 mean_color[0]   # B
#             ])

# Button to run extraction
# if dataset_path and os.path.isdir(dataset_path):
#     st.info("🔎 Scanning dataset...")
    
#     data = []
    
#     # Loop over files
#     for root, dirs, files in os.walk(dataset_path):
#         for file in files:
#             if file.lower().endswith(".tif"):  # process only thumbnail images
#                 file_path = os.path.join(root, file)
                
#                 img = cv2.imread(file_path)
#                 if img is None:
#                     continue
                
#                 # Extract features
#                 image_name = file
#                 height, width, channels = img.shape
#                 resolution = width * height
#                 mean_color = cv2.mean(img)[:3]  # (B, G, R)
                
#                 # Device name from folder name
#                 device_name = os.path.basename(root)
                
#                 # Append row
#                 data.append([
#                     image_name,
#                     device_name,
#                     resolution,
#                     width,
#                     height,
#                     channels,
#                     mean_color[2],  # R
#                     mean_color[1],  # G
#                     mean_color[0]   # B
#                 ])

# Convert to DataFrame
# df = pd.DataFrame(data, columns=[
#     "Image_Name", "Device_Name", "Resolution", 
#     "Width", "Height", "Channels", "Mean_R", "Mean_G", "Mean_B"
# ])


# # Folder for saving CSVs
# output_folder = r"D:\OneDrive\Desktop\Project_Trace_Finder\FeatureOutputs"

# # Create folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Save CSV inside FeatureOutputs
# output_path = os.path.join(output_folder, "image_features.csv")
# df.to_csv(output_path, index=False)

# print(f"✅ Features extracted and saved to {output_path}")


#    # Show preview
#    st.subheader("📊 Extracted Features (Preview)")
#    st.dataframe(df.head(20))
    
#    # Save CSV in same dataset folder
#     save_path = os.path.join(dataset_path, "image_features.csv")
#     df.to_csv(save_path, index=False)
#     st.success(f"✅ Features saved to {save_path}")
    
#     # Class distribution (Device_Name)
#     if "Device_Name" in df.columns:
#         st.subheader("📈 Device Distribution")
#         st.bar_chart(df["Device_Name"].value_counts())
    
#     # Show sample images
#     st.subheader("🖼️ Sample Images from Devices")
#     cols = st.columns(5)
#     for idx, cls in enumerate(df["Device_Name"].unique()):
#         # Find one sample from this class
#         sample_row = df[df["Device_Name"] == cls].iloc[0]
#         sample_path = os.path.join(dataset_path, cls, sample_row["Image_Name"])
#         if os.path.exists(sample_path):
#             img = Image.open(sample_path)
#             cols[idx % 5].image(img, caption=cls, use_container_width=True)
            
# elif dataset_path:
#     st.error("❌ Invalid dataset path. Please enter a valid folder.")



import streamlit as st
import os
import cv2
import pandas as pd
from PIL import Image
import io

# Streamlit page setup
st.set_page_config(page_title="Image Feature Extractor", layout="wide")
st.title("📸 Image Feature Extractor - Forgery Dataset Project")

# User input: dataset path
dataset_path = st.text_input("📂 Enter dataset folder path:", "")

# Button to run extraction
if dataset_path and os.path.isdir(dataset_path):
    st.info("🔎 Scanning dataset...")
    
    data = []
    
    # Loop over files
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(".tif"):  # process only thumbnail images
                file_path = os.path.join(root, file)
                
                img = cv2.imread(file_path)
                if img is None:
                    continue
                
                # Extract features
                image_name = file
                height, width, channels = img.shape
                resolution = width * height
                mean_color = cv2.mean(img)[:3]  # (B, G, R)
                
                # Device name from folder name
                device_name = os.path.basename(root)
                
                # Append row
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
    
    # Show preview
    st.subheader("📊 Extracted Features (Preview)")
    st.dataframe(df.head(10))
    
    # Save CSV in same dataset folder
    # output_folder = r"D:\OneDrive\Desktop\Project_Trace_Finder\FeatureOutputs\csv"
    # os.makedirs(output_folder, exist_ok=True)

    # save_path = os.path.join(output_folder, "image_features.csv")
    # df.to_csv(save_path, index=False)
    # st.success(f"✅ Features saved to {save_path}")

    

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="image_features.csv",
        mime="text/csv"
    )
    
    # Class distribution (Device_Name)
    if "Device_Name" in df.columns:
        st.subheader("📈 Device Distribution")
        st.bar_chart(df["Device_Name"].value_counts())
    
    # Show sample images
    st.subheader("🖼️ Sample Images from Devices")
    cols = st.columns(5)
    for idx, cls in enumerate(df["Device_Name"].unique()):
        # Find one sample from this class
        sample_row = df[df["Device_Name"] == cls].iloc[0]
        sample_path = os.path.join(dataset_path, cls, sample_row["Image_Name"])
        if os.path.exists(sample_path):
            img = Image.open(sample_path)
            # cols[idx % 5].image(img, caption=cls, use_container_width=True)
            cols[idx % 5].image(img, caption=cls, width="stretch")

elif dataset_path:
    st.error("❌ Invalid dataset path. Please enter a valid folder.")
