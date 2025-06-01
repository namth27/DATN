import os
import cv2
import numpy as np
from pipeline.yolov12 import detect_caries
from pipeline.process_crop_img import *
from pipeline.process_bboxs import *
from pipeline.heatmap import generate_heatmap
from pipeline.utils import *

def run_segmentation_pipeline(image_path, file_id):
    """
    - Nhận đường dẫn ảnh upload
    - Phát hiện vùng nghi sâu bằng YOLO
    - Dùng model 2 để phân đoạn các vùng đó
    - Trả về đường dẫn ảnh mask + overlay
    """
    # Đọc ảnh gốc (đảm bảo đọc được cả PNG RGBA)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn đã cho.")

    # Chuyển từ RGBA sang BGR nếu cần
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        
    # 1. Tạo heatmap
    heatmap_result = generate_heatmap(image)
    heatmap_path = f"static/results/{file_id}_heatmap.jpg"
    cv2.imwrite(heatmap_path, heatmap_result)
    
    heatmap_alpha_path = f"static/results/{file_id}_heatmap_alpha.png"
    save_heatmap_alpha(image, heatmap_alpha_path)

    # Phát hiện BB bằng YOLO
    detected_path = f"static/results/{file_id}_detected.jpg"
    bboxes = detect_caries(image_path, detected_path)
    if not bboxes:
        accuracy = 0
    else:
        confidences = [conf for (_, _, _, _, conf) in bboxes]
        accuracy = sum(confidences) / len(confidences)
        accuracy = round(accuracy * 100, 2)
        
    bbox_alpha_path = f"static/results/{file_id}_bbox_alpha.png"
    save_bbox_alpha(image, bboxes, bbox_alpha_path)
    
    # Nếu không có BB, trả ảnh gốc
    if not bboxes:
        return {
            "original": "/" + image_path,
            "mask": None,
            "overlay": "/" + image_path,
            "mask_path": None,
            "overlay_path": "/" + image_path
        }

    # Crop và resize để đưa vào model 2
    crops = crop_and_resize_for_model2(image, bboxes)
    final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for i, crop in enumerate(crops):
        pred_mask = process_crop_image(crop)  # Output 224x224
        pred_mask = (pred_mask * 255).astype(np.uint8)

        x1, y1, x2, y2, _ = bboxes[i]
        resized = cv2.resize(pred_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], resized)

    # Tạo overlay mờ
    green_overlay = np.zeros_like(image, dtype=np.uint8)
    green_overlay[:, :, 1] = 255  # Kênh G = xanh lá

    # Mức trong suốt (0.0 = hoàn toàn gốc, 1.0 = hoàn toàn màu overlay)
    alpha = 0.4  

    # Tạo overlay
    overlay = image.copy()
    mask_area = final_mask > 127
    if np.any(mask_area):
        overlay[mask_area] = cv2.addWeighted(
            image[mask_area], 1 - alpha,
            green_overlay[mask_area], alpha,
            0
        )
    # Lưu kết quả
    mask_path = f"static/results/{file_id}_mask.jpg"
    overlay_path = f"static/results/{file_id}_overlay.jpg"
    cv2.imwrite(mask_path, final_mask)
    cv2.imwrite(overlay_path, overlay)
    
    overlay_alpha_path = f"static/results/{file_id}_overlay_alpha.png"
    save_overlay_alpha(image, final_mask, overlay_alpha_path)
    
    diagnosis = "Không phát hiện sâu răng" if np.count_nonzero(final_mask) == 0 else "Phát hiện vùng nghi sâu răng"
    return {
        "original": "/" + image_path,
        "heatmap": "/" + heatmap_path,
        "bbox_path": "/" + detected_path,
        "mask_path": "/" + mask_path,
        "overlay_path": "/" + overlay_path,
        "heatmap_alpha": "/" + heatmap_alpha_path,
        "overlay_alpha": "/" + overlay_alpha_path,
        "bbox_alpha": "/" + bbox_alpha_path,
        "diagnosis": diagnosis,
        "accuracy": accuracy
    }