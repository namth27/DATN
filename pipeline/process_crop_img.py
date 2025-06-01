from torchvision import transforms
from PIL import Image
import torch
from pipeline.unet_transformer import *
import cv2
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MTUNet(2)
model.to(device)
checkpoint = torch.load(r"D:/Downloads/epoch_16.pth", map_location=device)
#checkpoint = torch.load(r"D:/Desktop/Code/python/dental_caries_segmentation/backend/models/unet_transformer.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()  


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def process_crop_image(image_np):
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image_pil).unsqueeze(0).to(device) # Thêm chiều batch

    with torch.no_grad():
        # Truyền hình ảnh qua mô hình
        output = model(image_tensor)  
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return predicted_mask


