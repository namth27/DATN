import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO

# Load YOLO model
model_path = "D:/Desktop/HỌC TẬP/DA/YOLO/v7/detect/train/weights/last.pt"

#model_path = "D:/Desktop/Code/python/dental_caries_segmentation/backend/models/yolov12.pt"  # Đường dẫn đến model YOLO
model = YOLO(model_path)

def detect_caries(image_path, output_path, conf_threshold=0.5):
    """
    Chạy model YOLO để nhận diện sâu răng trong ảnh.
    - Trả về danh sách bounding boxes.
    - Lưu ảnh có vẽ BBs vào thư mục `results/`.
    """
    # Load ảnh
    image = cv2.imread(image_path)
    results = model(image)[0]  # Chạy YOLO
    
    bboxes = []
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Lấy toạ độ BB
        confidence = float(result.conf[0])  # Độ tin cậy
        if confidence >= conf_threshold:
            bboxes.append((x1, y1, x2, y2, confidence))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Lưu ảnh kết quả
    cv2.imwrite(output_path, image)
    print(f"Ảnh đã được lưu tại: {output_path}")

    txt_output_path = os.path.splitext(output_path)[0] + "_bboxes.txt"
    with open(txt_output_path, 'w') as f:
        for bbox in bboxes:
            x1, y1, x2, y2, conf = bbox
            f.write(f"{x1} {y1} {x2} {y2} {conf:.4f}\n")
            
    return bboxes


def read_bboxes_from_txt(txt_path):
    """
    Đọc file .txt chứa bounding boxes có định dạng:
    x1 y1 x2 y2 confidence
    Trả về danh sách các BBs dạng tuple: (x1, y1, x2, y2, confidence)
    """
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                x1, y1, x2, y2 = map(int, parts[:4])
                confidence = float(parts[4])
                bboxes.append((x1, y1, x2, y2, confidence))
    return bboxes
