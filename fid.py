import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import inception_v3
from scipy.linalg import sqrtm


def get_inception_model(device):
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

def get_activations(dataloader, model, device):
    """
    使用给定模型（Inception v3）从dataloader中提取特征激活（2048维）
    """
    activations = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            # 前向传播提取特征
            preds = model(imgs)
            # 假设输出维度为 [batch_size, 2048]
            activations.append(preds.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    return activations

def calculate_statistics(activations):
    """
    根据提取的特征计算均值和协方差
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    计算FID分数
    """
    diff = mu1 - mu2
    diff_squared = diff.dot(diff)

    # 计算协方差矩阵的平方根
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


if __name__ == "__main__":
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_inception_model(device)

    # 请根据自己的数据路径修改
    real_image_dir = "E:/code/watermark/code/RG/data/3"
    fake_image_dir = "E:/code/watermark/code/RG/data/4"

    real_dataloader = preprocess_images(real_image_dir, device)
    fake_dataloader = preprocess_images(fake_image_dir, device)

    # 提取特征
    real_activations = get_activations(real_dataloader, model, device)
    fake_activations = get_activations(fake_dataloader, model, device)

    # 计算统计量
    mu_real, sigma_real = calculate_statistics(real_activations)
    mu_fake, sigma_fake = calculate_statistics(fake_activations)

    # 计算FID
    fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    print("FID score:", fid_score)