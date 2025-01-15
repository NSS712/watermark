import numpy as np
from utils import images_to_numpy, numpy_to_images
import torch
import torch.fft as fft

# def map(x, k=5):
#     if (x//k)%2 : # x属于红区
#         if x%k < k//2 : # 向下取整
#             return x - x%k - 1
#         else:
#             return (x//k + 1)*k
#     else:
#         return x

def modify_red_to_green(images, k=5, value_range=None):
    """
    将输入的图片,按照k个像素取值一组，将奇数组的像素全都映射到最近的偶数组。
    
    :param iamge: numpy类型的数组，可以是单张或者多组图片
    :param k: 每组的像素取值间隔
    :value_range: 需要映射为（0，255）的像素值，可以通过value_range参数指定 value_range=(0,255)
    """
    if value_range:
        pix_map = np.array(pixel_map(k=k, value_range=value_range))
    elif isinstance(images, torch.Tensor):
        device = images.device
        images = images.cpu().numpy()
        pix_map = np.vectorize(map)(images, k=k)
        return torch.from_numpy(pix_map).to(device)
    else:
        pix_map = np.vectorize(map)(images, k=k)
    return pix_map


class Watermark_k_layer:
    @staticmethod
    def modify_k_percents(images, k):
        """
        对图像进行频域水印嵌入。
        """
        # 将图像转换为 NumPy 数组并转为张量
        images = torch.from_numpy(images_to_numpy(images))

        # 计算 FFT，获取模值和相位
        images_fft = torch.fft.fft2(images)
        images_fft_shifted = torch.fft.fftshift(images_fft, dim=(-1, -2))
        images_fft_mag = torch.sqrt(images_fft_shifted.real**2 + images_fft_shifted.imag**2)
        images_fft_phase = torch.atan2(images_fft_shifted.imag, images_fft_shifted.real)
        # print("ori  ="*20)
        # print(images_fft_mag[0:2,0:2,0:2,0:2])
        is_green = np.vectorize(Watermark_k_layer.is_green)
        # print(is_green(images_fft_mag[0:2,0:2,0:2,0:2], k))
        # 修改模值（嵌入水印）
        map = np.vectorize(Watermark_k_layer.map)
        images_fft_mag = map(images_fft_mag.detach().numpy(), k)
        # print("modi  ="*20)
        # print(images_fft_mag[0:2,0:2,0:2,0:2])
        # print(is_green(images_fft_mag[0:2,0:2,0:2,0:2], k))
        images_fft_mag = torch.from_numpy(images_fft_mag)

        # 重新组合频谱
        images_fft_modified = images_fft_mag * (torch.cos(images_fft_phase) + 1j * torch.sin(images_fft_phase))
        images_fft_inverse_shifted = torch.fft.ifftshift(images_fft_modified, dim=(-1, -2))
        images_modified = torch.fft.ifft2(images_fft_inverse_shifted).real

        # 转回 NumPy 格式并返回
        return numpy_to_images(images_modified.detach().numpy())

    @staticmethod
    def z_check(images, k):
        """
        检测图像是否嵌入了水印。
        """
        # 将图像转换为 NumPy 数组并转为张量
        images = torch.from_numpy(images)

        # 计算 FFT 并提取模值
        X = torch.fft.fft2(images)
        X_shifted = torch.fft.fftshift(X, dim=(-1, -2))
        X_shifted_mag = torch.sqrt(X_shifted.real**2 + X_shifted.imag**2).detach().numpy()
        is_green = np.vectorize(Watermark_k_layer.is_green)
        # print("debug:\n",X_shifted_mag[0:2,0:2,0:2,0:2])
        # print(is_green(X_shifted_mag[0:2,0:2,0:2,0:2], k))
        # 检测水印的存在（统计是否符合嵌入规则）
        n = images.shape[1] * images.shape[2] * images.shape[3]

        watermark_scores = np.array([
            (np.sum(is_green(img, k))) / n
            for img in X_shifted_mag
        ])
        return watermark_scores

    @staticmethod
    def map(x, k):
        """
        对模值进行水印嵌入的核心逻辑。
        """
        base = k
        t = np.floor(x / base)
        if t % 2: # 红区
            if x % base < base / 2:
                return x - x % base - 0.5
            else:
                return np.ceil(x / base) * base + 0.5
        else:
            return x
    
    @staticmethod
    def scalling(x,k):
        base = Watermark_k_layer.base(x, k)
        base_2 = x - x % (2 * base)
        sc = base_2 + (x % (2 * base)) / 2
        return sc

    @staticmethod
    def is_green(x, k):
        """
        判断模值是否满足水印规则。
        """
        base = k
        t = np.floor(x / base)
        return t % 2 < 0.0001

    @staticmethod
    def log(x, base):
        """
        计算以 base 为底的对数。
        """
        return np.log(x) / np.log(base)

    @staticmethod
    def base(x, k):
        """
        计算当前模值的分段基准。
        """
        return k ** (np.floor(Watermark_k_layer.log(x, k)))


# class Watermark_k_layer:
#     @staticmethod
#     def modify_k_percents(images, k):
#         images = torch.from_numpy(images_to_numpy(images))
#         images_fft = fft.fftshift(fft.fft2(images), dim=(-1, -2))
#         images_fft_real, images_fft_imag = images_fft.real.detach().numpy() , images_fft.imag.detach()
#         images_fft_real = np.vectorize(Watermark_k_layer.map)(images_fft_real, k)
#         images_fft = torch.from_numpy(images_fft_real) + 1j*images_fft_imag
#         images = fft.ifft2(fft.ifftshift(images_fft, dim=(-1, -2))).real.detach().numpy()
#         return numpy_to_images(images)

#     def modify_k_percents_mod(images, k):
#         images = torch.from_numpy(images_to_numpy(images))
        
#         # 计算 FFT 并提取模值和相位
#         images_fft = torch.fft.fft2(images)
#         images_fft_shifted = torch.fft.fftshift(images_fft, dim=(-1, -2))
#         images_fft_mag = torch.sqrt(images_fft_shifted.real**2 + images_fft_shifted.imag**2)
#         images_fft_phase = torch.atan2(images_fft_shifted.imag, images_fft_shifted.real)
        
#         # 修改模值（嵌入水印）
#         images_fft_mag = np.vectorize(Watermark_k_layer.map)(images_fft_mag.detach().numpy(), k)
#         images_fft_mag = torch.from_numpy(images_fft_mag)
        
#         # 重新组合频谱
#         images_fft_modified = images_fft_mag * (torch.cos(images_fft_phase) + 1j * torch.sin(images_fft_phase))
#         images_fft_inverse_shifted = torch.fft.ifftshift(images_fft_modified, dim=(-1, -2))
#         images_modified = torch.fft.ifft2(images_fft_inverse_shifted).real
        
#         return numpy_to_images(images_modified.detach().numpy())

    
#     @staticmethod
#     def z_check(images, k):
#         images = torch.from_numpy(images_to_numpy(images))
#         X = fft.fft2(images)
#         X_shifted = fft.fftshift(X, dim=(-1, -2))
#         X_shifted_real = X_shifted.real.detach().numpy()
#         n = images.shape[1] * images.shape[2] * images.shape[3]
#         return np.array([abs(np.sum(Watermark_k_layer.is_green(np.abs(img), k))* 2 - n)/n for img in X_shifted_real])

#     @staticmethod
#     def map(x, k):
#         if x < 0:
#             return -Watermark_k_layer.map_pos(-x, k)
#         else:
#             return Watermark_k_layer.map_pos(x, k)
    
#     @staticmethod
#     def map_pos(x, k):
#         base = Watermark_k_layer.base(x, k)
#         t = np.floor(x // base)
#         return  x - x%base if t%2 else x

#     @staticmethod
#     def log(x, base):
#         return  np.log(x) / np.log(base)
    
#     @staticmethod
#     def base(x, k):
#         return k ** (np.floor(Watermark_k_layer.log(x, k)))
    
#     @staticmethod
#     def is_green(x, k):
#         base = Watermark_k_layer.base(x, k)
#         t = (x // base).astype(np.int)
#         return t%2 == 0
        

def add_wm_FFT(images,k = 5):
    images = torch.from_numpy(images_to_numpy(images))
    X = fft.fft2(images)
    # X1 = X[0].detach().numpy()
    # X1 = np.log(np.where(X1 <= 0, 1e-10, X1))
    X_shifted = fft.fftshift(X, dim=(-1, -2))
    X_imag = X_shifted.imag
    X_shifted_modi = modify_red_to_green(X_shifted.real, k=k)
    X_shifted = X_shifted_modi + 1j*X_imag
    # X2 = X_shifted[0].detach().numpy()
    # X2 = np.log(np.where(X2 <= 0, 1e-10, X2))
    X_ishifted = fft.ifftshift(X_shifted, dim=(-1, -2))
    y = fft.ifft2(X_ishifted)  # [1, 3, H, W], 复数
    #print(y[0][0])
    y = y.real.detach().numpy()
    return numpy_to_images(y)


def map_255(x, k=5):
        if (x//k)%2 : # x属于红区
            if x%k < k//2 or (x//k + 1)*k > 255: # 向下取整
                return max(0, x - x%k - 1)
            else:
                return (x//k + 1)*k
        else:
            return x
    

def pixel_map(k=5, value_range=(0,255)):
    '''
    映射像素值，以k为分组，对x执行分区映射
    x 取值为0-255
    '''
        
    return [map_255(x,k=k) for x in list(range(value_range[0], value_range[1]+1))]

# def add_k(images, k=5):
#     images = images_to_numpy(images)
#     n = images.shape[0]//2
#     images_ori = images[n:].copy()
#     images_watermark = modify_red_to_green(images[n:], k=k)
    
    
    
if __name__=="__main__":
    # a = np.array(list(range(-15,15)))
    # print(a)
    # k = 5
    # # print(modify_red_to_green(a,k))
    # ans = [map(x,k=k) for x in a]    
    # for i in range(len(a)):
    #     print(f"{a[i]}->{ans[i]}")
    print("start    ")
    x = torch.Tensor([[[[1,2,3],[4,5,6],[7,8,9]]]])
    print(modify_red_to_green(x, k=5))