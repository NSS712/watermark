import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

def get_inception_model(device='cuda'):
    # 加载预训练的Inception v3模型
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    # 去掉Inception的fc层，只保留到最后的平均池化层的输出
    inception.fc = nn.Identity()
    inception.eval()
    return inception

def preprocess_images(image_dir, device, batch_size=32, num_workers=0):
    """
    创建DataLoader，从文件夹中读取图像，预处理后输出张量
    """

    # Inception v3 expects images of size (299,299)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # ImageNet mean/std
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = torchvision.datasets.ImageFolder(
        root=image_dir,
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader

def fid():
    model = get_inception_model()
    
    