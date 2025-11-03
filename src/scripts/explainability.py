# D:\Project_Trace_Finder\src\scripts\explainability.py

import tensorflow as tf
import keras
import numpy as np
import cv2

def _find_layer(model, layer_names_list):
    """Internal function to find the first valid layer from a list of names."""
    for name in layer_names_list:
        try:
            return model.get_layer(name=name)
        except ValueError:
            continue # Try the next name
    
    # If none are found, raise an error
    raise ValueError(f"Could not find any of the target layers. Tried: {layer_names_list}. Full list: {[l.name for l in model.layers]}")

def make_gradcam_heatmap(img_array, feature_array, model, last_conv_layer_names, pred_index=None):
    """
    Creates a Grad-CAM heatmap for a hybrid (two-input) model.
    'last_conv_layer_names' should be a list of possible names to try.
    """
    # Find the correct layer from the list
    target_layer = _find_layer(model, last_conv_layer_names)

    # Create a model that maps the inputs to the last conv layer output and the final predictions
    grad_model = keras.models.Model(
        inputs=[model.inputs[0], model.inputs[1]], # [img_input, feat_input]
        outputs=[target_layer.output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds_list = grad_model([img_array, feature_array])
        preds_tensor = preds_list[0]
        if pred_index is None:
            pred_index = tf.argmax(preds_tensor[0])
        class_channel = preds_tensor[:, pred_index]

    # Gradient of the output neuron w.r.t. the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def get_superimposed_image(img_path, heatmap, alpha=0.5):
    """
    Superimposes the Grad-CAM heatmap onto the original image.
    """
    img = cv2.imread(img_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap_rgb = np.uint8(255 * heatmap_resized)
    heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    
    superimposed_img = (heatmap_rgb * alpha + img * (1 - alpha)).astype(np.uint8)
    
    return superimposed_img, heatmap_rgb





# # D:\Project_Trace_Finder\src\scripts\explainability.py

# import tensorflow as tf
# import keras
# import numpy as np
# import cv2

# def make_gradcam_heatmap(img_array, feature_array, model, last_conv_layer_name, pred_index=None):
#     """
#     Creates a Grad-CAM heatmap for a hybrid (two-input) model.
#     """
#     # Create a model that maps the inputs to the last conv layer output and the final predictions
#     grad_model = keras.models.Model(
#         inputs=[model.inputs[0], model.inputs[1]], # [img_input, feat_input]
#         outputs=[model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     # Compute gradients
#     with tf.GradientTape() as tape:
#         # We need to provide both inputs
#         last_conv_layer_output, preds = grad_model([img_array, feature_array])
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     # Gradient of the output neuron w.r.t. the output feature map of the last conv layer
#     grads = tape.gradient(class_channel, last_conv_layer_output)

#     # Vector of mean intensity of the gradient over a specific feature map channel
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     # Multiply each channel in the feature map by "how important this channel is"
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # Normalize the heatmap
#     heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
#     return heatmap.numpy()

# def get_superimposed_image(img_path, heatmap, alpha=0.5):
#     """
#     Superimposes the Grad-CAM heatmap onto the original image.
#     """
#     # Load the original, unprocessed image
#     img = cv2.imread(img_path)
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Resize heatmap to match original image size
#     heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
#     # Convert heatmap to RGB
#     heatmap_rgb = np.uint8(255 * heatmap_resized)
#     heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
#     heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    
#     # Superimpose
#     superimposed_img = (heatmap_rgb * alpha + img * (1 - alpha)).astype(np.uint8)
    
#     return superimposed_img, heatmap_rgb