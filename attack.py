from PIL import Image
import random
import numpy as np
import math
from examine import z_check
from sklearn.metrics import roc_curve, roc_auc_score
import time
from utils import *
from scipy.interpolate import interp1d

def rotate_image(images, angle=None, expand=True, type="RGB"):
    """
    对图片进行旋转变换。
    
    :param image_path: 图片路径
    :param angle: 旋转角度，默认为随机生成角度
    :param expand: 是否扩大图片尺寸以适应旋转后的完整图像，默认 True
    :return: 旋转后的图片对象
    """
    # 打开图片
    images = numpy_to_images(images, type=type)

    # 如果未指定角度，随机生成一个角度
    if angle is None:
        angle = random.uniform(0, 360)  # 随机角度 [0, 360)

    # 旋转图片
    
    ans = [img.rotate(angle, expand=expand) for img in images]

    return images_to_numpy(ans)

def random_crop(images, area_ratio, type="RGB"):
    """
    对输入图像按面积比例随机裁剪，并调整回原尺寸。

    :param image: Pillow 的 Image 对象
    :param area_ratio: 裁剪区域面积占原图面积的比例 (0, 1]
    :return: 裁剪并调整后的 Image 对象
    """
    # 获取图像的原始宽高
    images = numpy_to_images(images, type=type)
    img_width, img_height = images[0].size

    # 检查面积比例是否合法
    if area_ratio <= 0 or area_ratio > 1:
        raise ValueError("面积比例必须在 (0, 1] 范围内")

    # 计算裁剪区域的目标面积
    target_area = img_width * img_height * area_ratio

    # 随机生成裁剪区域的宽高（保持接近原图宽高比）
    aspect_ratio = random.uniform(0.5, 2.0)  # 宽高比随机生成
    crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
    crop_height = int(round(math.sqrt(target_area / aspect_ratio)))

    # 确保裁剪区域不超过原图尺寸
    crop_width = min(crop_width, img_width)
    crop_height = min(crop_height, img_height)

    # 随机生成裁剪区域的左上角坐标
    left = random.randint(0, img_width - crop_width)
    top = random.randint(0, img_height - crop_height)

    # 计算裁剪区域的右下角坐标
    right = left + crop_width
    bottom = top + crop_height

    # 裁剪图像

    cropped_images = [image.crop((left, top, right, bottom)) for image in images]

    # 调整裁剪后的图像回到原始尺寸
    resized_images = [cropped_image.resize((img_width, img_height), Image.BICUBIC).convert(type) for cropped_image in cropped_images]

    return images_to_numpy(resized_images)

def compression(images, quality=50, type="RGB"):
    images = numpy_to_images(images, type=type)
    images_com = []
    current_timestamp = int(time.time())
    for i, img in enumerate(images):
        img.save(f"./data/com/{current_timestamp}_compressed_quality_{i}_{quality}.jpg", "JPEG", quality=quality)
        img_com = Image.open(f"./data/com/{current_timestamp}_compressed_quality_{i}_{quality}.jpg").convert(type)
        images_com.append(img_com)
    return images_to_numpy(images_com)

def attack_all(images,k,target_fpr=0.1, type="RGB"):
    # 攻击
    images = images_to_numpy(images)
    images_crop = random_crop(images,area_ratio=0.75,type=type)
    images_compressed = compression(images,quality=25, type=type)
    # 计算z检验
    z_watermark = z_check(images,k=k)
    z_crop = z_check(images_crop,k=k)
    z_compressed = z_check(images_compressed,k=k)
    n = images.shape[0]//2
    y_true = np.array([1] * n + [0] * n)
    
    results = []
    for z_values in [z_watermark, z_crop, z_compressed]:
        fpr, tpr, thresholds = roc_curve(y_true, z_values)
        interp_tpr = interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")
        tpr_at_target_fpr = interp_tpr(target_fpr)
        auc = roc_auc_score(y_true, z_values)
        results.append((auc, tpr_at_target_fpr))
    return results