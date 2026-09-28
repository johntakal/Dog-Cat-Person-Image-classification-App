import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


class ConvNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class ImageClassifier:
    def __init__(self, model_path=None, class_name=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        if class_name is None:
            self.class_name = {0: 'Cat', 1: 'Dog', 2: 'Person'}
        else:
            self.class_name = class_name
            
        self.model = ConvNet(num_classes=len(self.class_name))
        
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model path not found or not provided: {model_path}")

        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            base_dir = os.path.dirname(image_path) or '.'
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
            base_dir = '.'
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
        label = self.class_name.get(pred_idx, f"Class {pred_idx}")
        result_text = f"{label} ({confidence * 100:.1f}%)"
        
        # Create visually labeled output image
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        
        # Determine banner & font sizing
        font_size = max(16, int(image.height * 0.05))
        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()
            
        banner_text = f"{label}: {confidence * 100:.1f}%"
        bbox = draw.textbbox((0, 0), banner_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        padding = 10
        banner_h = text_h + padding * 2
        banner_w = max(image.width, text_w + padding * 2)
        
        draw.rectangle([(0, 0), (banner_w, banner_h)], fill=(18, 24, 38))
        draw.text((padding, padding), banner_text, fill=(46, 204, 113), font=font)
        
        output_path = os.path.join(base_dir, 'output_labeled_image.jpg')
        output_image.save(output_path)
        
        return result_text, output_path
