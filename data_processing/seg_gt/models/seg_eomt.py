"""
EoMT Cityscapes semantic segmentation wrapper.

Uses HuggingFace Transformers API with DINOv2-L backbone.
19 Cityscapes driving classes, runs at full resolution.
"""
import torch
import numpy as np
from PIL import Image


EOMT_MODEL_ID = "tue-mps/cityscapes_semantic_eomt_large_1024"


def load_model(device="cuda"):
    """Load EoMT from HuggingFace."""
    from transformers import EomtForUniversalSegmentation, AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(EOMT_MODEL_ID)
    model = EomtForUniversalSegmentation.from_pretrained(EOMT_MODEL_ID)
    model = model.to(device).eval()
    return model, processor


def run_inference(model, processor, img_bgr, device="cuda"):
    """Run EoMT semantic segmentation.

    Args:
        model: loaded EoMT model
        processor: HuggingFace image processor
        img_bgr: (H, W, 3) uint8 BGR CLAHE'd frame

    Returns:
        semantic: (H, W) uint8 Cityscapes class IDs (0-18)
    """
    import cv2

    H, W = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # BatchFeature.to(device) moves all tensors to the device (non-tensors untouched).
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(H, W)]
    )[0]
    return result.cpu().numpy().astype(np.uint8)
