import numpy as np

def create_overlay_mask(image_shape, mask_crop, bbox): 
    """
    Tạo mask có kích thước bằng ảnh gốc, gán vùng mask từ crop vào đúng vị trí bbox.
    """
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask_crop
    return full_mask
