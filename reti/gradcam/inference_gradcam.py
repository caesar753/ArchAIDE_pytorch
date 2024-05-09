#importing libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn


from torchvision import models
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import argparse

import copy
import os

import numpy as np
import cv2

import torch
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image

import matplotlib.pyplot as plt
from datetime import datetime

from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# infer_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='Torch device to use')
    parser.add_argument(
        '--model',
        type=str,
        default='..\\Data\\Trained_models\\202404271957_resnet50_200epochs_on_1000epochs_5batch_0.001LR_0.6dropout_sgdoptimizer_0.485_accuracy_top1_0.777_accuracy_top5.pth',
        help='Input image path'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation\
        \n ')
    

    return args


def infer_transform (image):
    trans_img = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])
    img = trans_img(image).unsqueeze(0)

    return img

if __name__ == '__main__':

    # infer_transform= transforms.Compose([
    #     transforms.Resize((224,224)),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    #     transforms.Normalize(mean = [0.485, 0.456, 0.406],
    #                          std = [0.229, 0.224, 0.225])
    # ])

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    model = torch.load(os.path.join("..\\..\\Data\\Trained_models", args.model))
    # print(model)

    if args.device:
        model = model.to(torch.device(args.device)).eval()
    else:
        model = model.to(torch.device("cpu")).eval()

    image_dir = os.path.join(args.output_dir, f'{args.image_path[args.image_path.find("MTL"):]}')
    os.makedirs(image_dir, exist_ok=True)

    model = torch.load(os.path.join("..\\..\\Data\\Trained_models\\", args.model))
    # print(model)

    if args.device:
        model = model.to(torch.device(args.device)).eval()
    else:
        model = model.to(torch.device("cpu")).eval()

    image_dir = os.path.join(args.output_dir,f'{args.model}', f'{args.image_path[args.image_path.find("MTL"):]}')
    os.makedirs(image_dir, exist_ok=True)

    # if args.device:
    #     model = models.resnet101(pretrained=True).to(torch.device(args.device)).eval()
    # else:
    #     model = models.resnet101(pretrained=True).to(torch.device("cpu")).eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    target_layers = [model.layer4]
    # target_layers = [model.fc[-1]]
    # target_layers = [model.avgpool]
    # print(target_layers)

    image = Image.open(args.image_path)
    image = infer_transform(image).to(args.device)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(args.device)      

               
                        
    # print(input_tensor.size())

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]
        # print(grayscale_cam)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model = model, use_cuda = None)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cam_output_path = os.path.join(image_dir, f'{args.method}_cam.jpg')
    gb_output_path = os.path.join(image_dir, f'{args.method}_gb.jpg')
    cam_gb_output_path = os.path.join(image_dir, f'{args.method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)

    output = model(image)
    # output = model(input_tensor)
    # print(output)

    #Only prediction
    # prediction = int(torch.max(output.data, 1)[1].cpu().numpy())

    #Prediction with confidence level
    probs = nn.functional.softmax(output, dim=1)
    # print(probs)
    confidence = (torch.max(probs.data, 1))[0].cpu().numpy()
    prediction = (torch.max(probs.data, 1))[1].cpu().numpy()
    print('The prediction is %d with a confidence level of %.2f %%' % (prediction, (100* confidence)))