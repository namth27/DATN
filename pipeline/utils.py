import numpy as np
import cv2

def save_overlay_alpha(image, final_mask, output_path):
    overlay = np.zeros((*image.shape[:2], 4), dtype=np.uint8)  # RGBA

    # Màu overlay: xanh lá + alpha
    overlay[final_mask > 127] = [0, 255, 0, 100]  # (R,G,B,A)

    cv2.imwrite(output_path, overlay)

def save_heatmap_alpha(image, output_path):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_component = lab[:, :, 0]
    _, th = cv2.threshold(l_component, 127, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(th, (11, 11), 0)
    heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

    # Tạo ảnh RGBA từ heatmap
    heatmap_rgba = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
    heatmap_rgba[:, :, 3] = blur  # Dùng blur làm alpha

    cv2.imwrite(output_path, heatmap_rgba)

def save_bbox_alpha(image, bboxes, output_path):
    h, w = image.shape[:2]  # lấy chiều cao và chiều rộng từ ảnh gốc
    bbox_img = np.zeros((h, w, 4), dtype=np.uint8)  # ảnh RGBA

    for x1, y1, x2, y2, _ in bboxes:
        # Vẽ khung màu đỏ có alpha 100%
        cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (255, 0, 0, 255), 2)

    cv2.imwrite(output_path, bbox_img)

