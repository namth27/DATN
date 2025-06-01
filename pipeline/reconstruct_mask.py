import cv2
import numpy as np

def reconstruct_mask(image, mask_pred, bbox):
    """
    image: ảnh gốc (HxWx3)
    mask_pred: ảnh mask từ model2 (224x224)
    bbox: (x1, y1, x2, y2)
    """
    H, W, _ = image.shape
    x1, y1, x2, y2 = bbox
    crop_w, crop_h = x2 - x1, y2 - y1

    # Resize mask về lại kích thước vùng crop
    mask_resized = cv2.resize(mask_pred, (crop_w, crop_h))

    # Tạo mask toàn ảnh
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask_resized
    
    return full_mask

