import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import *

def z_check(images, k=5):
    n = images[0].shape[1] * images[0].shape[2] * images[0].shape[0]
    return np.array([abs(np.sum((img//k)%2==0)* 2 - n)/n for img in images])

def caculate(images_o, images_w):
    ssim = ssim_score(images_o, images_w)
    psnr = calculate_mean_psnr(images_o, images_w)
    return (ssim, psnr)
    

def ssim_score(images_ori, images_watermark):
    ans = []
    # images_ori = numpy_to_images(images_ori)
    # images_watermark = numpy_to_images(images_watermark)
    for i in range(len(images_ori)):
        score, _ = ssim(np.array(images_ori[i].convert('L')), np.array(images_watermark[i].convert('L')), full=True)
        ans.append(score)
    return np.array(ans).mean()

def calculate_mean_psnr(images1, images2):
    images1, images2 = images_to_numpy(images1), images_to_numpy(images2)
    if images1.shape != images2.shape:
        raise ValueError("The dimensions of the two image batches do not match!")
    
    mse = np.mean((images1 - images2) ** 2)
    if mse == 0:
        return float('inf')  # MSE 为零，PSNR 无穷大
    
    max_pixel = 255.0  # 假设输入为 8 位图像
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    return psnr