import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.Res import resnet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from datetime import datetime
import random

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9988))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


current_time = datetime.now().strftime("%m%d%H%M%S")


cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"---------The device available now is {device}---------")


model = resnet18(num_classes=10)
state_dict = torch.load(
    "/home/ubuntu/stamp_ln/pre-train/models/ResNet18_10.pt",
    map_location=device
)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
print(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")


if len(missing_keys) + len(unexpected_keys) > 0:
    print("There are missing BN stats, Not a source model, bn stats are missing!")
    with torch.no_grad():
        dummy_loader = DataLoader(torch.randn(100, 3, 32, 32), batch_size=8)
        model.train()
        for dummy_input in dummy_loader:
            _ = model(dummy_input)
        model.eval()

model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) 训练阶段没使用归一化 下面的反归一化也要同步注释
])


dataset = CIFAR10(root="/mnt/d/stamp_lib/datasets", train=False, download=True, transform=transform) # 
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # 设置 shuffle=False 以确保每次加载的顺序一致

images, labels = next(iter(dataloader))

images = images.to(device)
labels = labels.to(device)



target_layer = model.conv5_x[-1].residual_function[3] # -2 -> 3

# Grad-CAM
cam = GradCAM(model=model, target_layers=[target_layer])  

fig, axes = plt.subplots(8, 8, figsize=(24, 24))  
fig.subplots_adjust(wspace=0.2, hspace=0.5)  


for i in range(32):
    input_tensor = images[i].unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)


    rgb_image = np.transpose(images[i].cpu().numpy(), (1, 2, 0))  # 转换为 (32, 32, 3)
    # rgb_image = (rgb_image * np.array([0.2023, 0.1994, 0.2010])) + np.array([0.4914, 0.4822, 0.4465])  # 反归一化
    rgb_image = np.clip(rgb_image, 0, 1)

    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)


    output = model(input_tensor)
    predicted_category = output.argmax(dim=1).item() # dim= 1 


    row = i // 4  # 8 行
    col = (i % 4) * 2  # 每对图占两列

    ax = axes[row, col]
    ax.imshow(rgb_image)
    ax.axis('off')
    ax.set_title(f"True: {cifar10_classes[labels[i]]}", fontsize=22)  

    ax = axes[row, col + 1]
    ax.imshow(visualization)
    ax.axis('off')
    ax.set_title(f"Pred: {cifar10_classes[predicted_category]}", fontsize=22) 


for i in range(32, 64):
    row = i // 8
    col = i % 8
    axes[row, col].axis('off')


plt.tight_layout()
plt.savefig(f"{current_time}_batch_gradcam.jpg", bbox_inches='tight', dpi=300)  
print("Batch Grad-CAM image saved")