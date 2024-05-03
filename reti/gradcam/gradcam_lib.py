import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from pytorch_grad_cam import GradCAM

# Load pre-trained model
model = models.resnet50(pretrained=True)


# Preprocessing transform
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# save_path = "..\\Data\\Trained_models\\"

# to_load = input("Enter the model to load \n")

# load_model = (str(save_path) + str(to_load))
# model = torch.load(load_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
  model = model.cuda()
#   model.to(device)

model.eval()

# Load image
to_infer = input("Insert path where is located the image to predict\n")
image = Image.open(to_infer)
image_tensor = infer_transform(image).unsqueeze(0)

# Instantiate GradCAM and set target layer
target_layer = model.layer4[-1]
grad_cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)


# Generate Grad-CAM heatmap
heatmap = grad_cam(input_tensor=image_tensor, target_category=None)

# Overlay heatmap on the image
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.imshow(image, alpha=0.5)
plt.axis('off')
plt.show()