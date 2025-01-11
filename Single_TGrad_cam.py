from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.models.Res import resnet18
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime

current_time = datetime.now().strftime("%m%d%H%M%S")

cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

model = resnet18(num_classes=10)

state_dict = torch.load(
    "/home/ubuntu/stamp_ln/pre-train/models/ResNet18_10.pt"
    , map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
print(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

if len(missing_keys) + len(unexpected_keys) > 0:
    print("There are missing keys, Not a source model, bn params are missing!")
    # Fix BatchNorm stats
    with torch.no_grad():
        dummy_data = torch.randn(8, 3, 32, 32)
        model.train()
        _ = model(dummy_data)
    model.eval()



transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

image_path = "/home/ubuntu/stamp_ln/0027.jpg" 
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Verify target layer
target_layer = model.conv5_x[-1].residual_function[-2]


if len(missing_keys) + len(unexpected_keys) > 0:
    print("There are missing BN stats, Not a source model, bn stats are missing!")
    # Update BatchNorm statistics
    with torch.no_grad():
        dummy_loader = torch.utils.data.DataLoader(
            torch.randn(100, 3, 32, 32),  # Replace with your input dimensions
            batch_size=8,
        )
        model.train()  # Switch to training mode for BatchNorm update
        for dummy_input in dummy_loader:
            dummy_input = dummy_input.to(next(model.parameters()).device)
            _ = model(dummy_input)
        model.eval()  # Switch back to evaluation mode


# Grad-CAM
cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

rgb_image = np.array(image) / 255.0
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

output = model(input_tensor)
predicted_category = output.argmax().item()
print(f"Predicted category: {predicted_category} ({cifar10_classes[predicted_category]})")


fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(rgb_image)
ax[0].axis('off')
ax[0].set_title("Original Image", pad=20)

ax[1].imshow(visualization)
ax[1].axis('off')
ax[1].set_title(f"Grad-CAM: {cifar10_classes[predicted_category]}", pad=20)


plt.subplots_adjust(wspace=0.05, top=0.85)

plt.tight_layout()
plt.savefig(f"{current_time}.jpg")
print("Grad-CAM image saved")