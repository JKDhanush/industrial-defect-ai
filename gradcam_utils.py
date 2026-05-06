import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# GENERATE GRADCAM
# -----------------------------

def generate_gradcam(model, input_tensor, image_np):

    # -----------------------------
    # RESIZE IMAGE TO MATCH MODEL INPUT
    # -----------------------------

    image_np = cv2.resize(image_np, (224, 224))

    # -----------------------------
    # TARGET LAYER
    # -----------------------------

    target_layers = [model.layer4[-1]]

    # -----------------------------
    # CREATE GRADCAM OBJECT
    # -----------------------------

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # -----------------------------
    # GENERATE HEATMAP
    # -----------------------------

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    # -----------------------------
    # NORMALIZE IMAGE
    # -----------------------------

    image_np = image_np.astype(np.float32) / 255.0

    # -----------------------------
    # OVERLAY HEATMAP
    # -----------------------------

    visualization = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb=True
    )

    return visualization