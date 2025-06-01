import cv2
import numpy as np

def generate_heatmap(image: np.ndarray) -> np.ndarray:
    """
    Sinh heatmap từ ảnh RGB/BGR gốc bằng cách xử lý kênh L trong không gian LAB.
    Trả về ảnh đã overlay heatmap lên ảnh gốc.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_component = lab[:, :, 0]
    _, th = cv2.threshold(l_component, 127, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(th, (11, 11), 0)
    heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    result = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return result