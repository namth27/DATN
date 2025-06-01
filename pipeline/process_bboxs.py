import cv2
import numpy as np

def crop_and_resize_for_model2(image, bboxes, target_size=(224, 224)):
    """
    Cắt và resize các vùng chứa sâu răng sao cho phù hợp với đầu vào model 2.
    """
    H, W, _ = image.shape
    resized_images = []
    for x1, y1, x2, y2, conf in bboxes:
        # Cắt vùng ảnh dựa trên bounding box
        cropped_image = image[y1:y2, x1:x2]
        
        # Resize ảnh cropped về kích thước 224x224
        resized_image = cv2.resize(cropped_image, target_size)
        
        resized_images.append(resized_image)
    
    return resized_images




