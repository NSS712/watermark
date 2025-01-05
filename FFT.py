import torch
import torch.fft as fft
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as ssim
from add_watermark import modify_red_to_green

def calculate_psnr(img1, img2):
    # img1, img2: [H, W, C] 取值范围 0-255，类型可为 float32 或 uint8 均可
    # 注意确保两幅图大小、通道都一致
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    if mse == 0:
        return 100
    psnr_val = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr_val

def eval(img1, img2):  
    psnr_val = calculate_psnr(img1, img2)
    ssim_val = ssim(img1, img2, channel_axis=-1)
    return psnr_val, ssim_val




image_path = 'data/1/k/0.png'
img_pil = Image.open(image_path).convert('RGB')
img_np = np.array(img_pil, dtype=np.float32)  # (H, W, C)
x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, H, W]

# 2. 对图像进行 FFT 变换并移位到中心
X = fft.fft2(x)
X_shifted = fft.fftshift(X, dim=(-2, -1))
print(X_shifted.size()) #[1, 3, H, W]

X_shifted = modify_red_to_green(X_shifted.real, k=2)

# 3. 逆变换回到空间域
X_ishifted = fft.ifftshift(X_shifted, dim=(-2, -1))
y = fft.ifft2(X_ishifted)  # [1, 3, H, W], 复数

# 4. 只取实部并转换成可显示的格式
y_real = y.real
y_real_np = y_real.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]

# 假设原图像像素范围就是 [0,255]，为与原图一致，这里做 clip + uint8
y_real_np = np.clip(y_real_np, 0, 255).astype(np.uint8)
img_origin_uint8 = img_np.astype(np.uint8)


# 计算 SSIM 和 PSNR
psnr_val, ssim_val = eval(img_origin_uint8, y_real_np)
print(f"PSNR: {psnr_val:.4f} dB;SSIM: {ssim_val:.4f}")





# ==========================
#     可视化对比
# ==========================
y_real_np_flatten = y_real_np.flatten()
img_origin_uint8_flatten = img_origin_uint8.flatten()
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].hist(img_origin_uint8_flatten, bins=50, alpha=0.7, color='tab:blue')
axs[0].set_title('Original')
axs[0].set_xlabel('Pixel Intensity')
axs[0].set_ylabel('Count')

axs[1].hist(y_real_np_flatten, bins=50, alpha=0.7, color='tab:red')
axs[1].set_title('Reconstructed')
axs[1].set_xlabel('Pixel Intensity')
plt.tight_layout()
plt.savefig('./pic/fft_histogram.png')


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示原图
axes[0].imshow(img_origin_uint8)
axes[0].set_title('Original Image')
axes[0].axis('off')

# 显示重建图像
axes[1].imshow(y_real_np)
axes[1].set_title('Reconstructed Image')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('./pic/fft_reconstruction.png')
plt.show()