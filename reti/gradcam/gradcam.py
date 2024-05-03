import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Register hooks for gradient extraction
target_layer = model.layer4[-1]
feature_grad = None
def hook_fn(module, grad_in, grad_out):
    global feature_grad
    feature_grad = grad_out[0]

hook_handle = target_layer.register_backward_hook(hook_fn)

# Preprocessing transform
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image
to_infer = input("Insert path where is located the image to predict\n")
image = Image.open(to_infer)
image_tensor = infer_transform(image).unsqueeze(0)

# Forward pass
output = model(image_tensor)
predicted_class = torch.argmax(output).item()

# Backward pass to get gradients
model.zero_grad()
output[0, predicted_class].backward()
gradient = feature_grad.mean(dim=[2, 3], keepdim=True)
activation = target_layer.forward(image_tensor)

# Compute Grad-CAM heatmap
grad_cam = torch.mul(activation, gradient).sum(dim=1, keepdim=True)
grad_cam = nn.functional.relu(grad_cam)

# Upsample to image size
upsampled_grad_cam = nn.functional.interpolate(grad_cam, size=(image.height, image.width), mode="bilinear", align_corners=False)

# Convert to numpy array
grad_cam_np = upsampled_grad_cam.squeeze().cpu().numpy()
grad_cam_np = np.maximum(grad_cam_np, 0) / grad_cam_np.max()

# Overlay Grad-CAM heatmap on the image
heatmap = plt.imshow(grad_cam_np, cmap='jet', alpha=0.5)
plt.imshow(image, alpha=0.5)
plt.axis('off')
plt.show()

# Remove hook
hook_handle.remove()