from PIL import Image
import random
import numpy as np
import math

def numpy_to_images(numpy_array):
    """
    将形状为 (n, 3, 224, 224) 的 NumPy 数组转换为 n 张图片的 Image 对象。

    :param numpy_array: NumPy 数组，形状为 (n, 3, 224, 224)
    :return: 包含 n 张 Image 对象的列表
    """
    # 检查输入形状
    if len(numpy_array.shape) != 4 or numpy_array.shape[1] != 3:
        raise ValueError("输入数组形状必须为 (n, 3, height, width)")

    # 转换通道维度到最后，形状变为 (n, height, width, 3)
    numpy_array = np.transpose(numpy_array, (0, 2, 3, 1))

    # 确保像素值范围在 [0, 255]（如有需要可归一化或调整范围）
    numpy_array = np.clip(numpy_array, 0, 255).astype(np.uint8)

    # 转换为 Pillow 的 Image 对象
    images = [Image.fromarray(img) for img in numpy_array]
    
    return images

def images_to_numpy(images):
    return np.array([np.array(img) for img in images]).transpose(0,3,1,2)
    
def rotate_image(images, angle=None, expand=True):
    """
    对图片进行旋转变换。
    
    :param image_path: 图片路径
    :param angle: 旋转角度，默认为随机生成角度
    :param expand: 是否扩大图片尺寸以适应旋转后的完整图像，默认 True
    :return: 旋转后的图片对象
    """
    # 打开图片
    images = numpy_to_images(images)

    # 如果未指定角度，随机生成一个角度
    if angle is None:
        angle = random.uniform(0, 360)  # 随机角度 [0, 360)

    # 旋转图片
    
    ans = [img.rotate(angle, expand=expand) for img in images]

    return images_to_numpy(ans)

def random_crop(images, area_ratio):
    """
    对输入图像按面积比例随机裁剪，并调整回原尺寸。

    :param image: Pillow 的 Image 对象
    :param area_ratio: 裁剪区域面积占原图面积的比例 (0, 1]
    :return: 裁剪并调整后的 Image 对象
    """
    # 获取图像的原始宽高
    images = numpy_to_images(images)
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
    resized_images = [cropped_image.resize((img_width, img_height), Image.BICUBIC) for cropped_image in cropped_images]

    return images_to_numpy(resized_images)

def compression(images, quality=50):
    images = numpy_to_images(images)
    images_com = []
    for i, img in enumerate(images):
        img.save(f"./data/com/compressed_quality_{i}_{quality}.jpg", "JPEG", quality=quality)
        img_com = Image.open(f"./data/com/compressed_quality_{i}_{quality}.jpg").convert("RGB")
        images_com.append(img_com)
        # if i == 0:
        #     a = np.array(img).flatten()
        #     b = np.array(img_com).flatten()
        #     for i in range(len(a)):
        #         print(f'{a[i]} -> {b[i]}')
    return images_to_numpy(images_com)
        