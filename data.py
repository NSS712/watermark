import os
import random
from PIL import Image
import numpy as np

def load_random_images(directory, num_images=1000, target_size=(224,224), type="RGB"):
    """
    从目录中随机选择 num_images 张图片并加载到内存。
    
    :param directory: 图片所在的目录路径
    :param num_images: 随机选择的图片数量，默认为1000
    :return: 加载的图片列表
    """
    image_files = [os.path.join(directory, f) for f in os.listdir(directory)]
    
    if num_images != -1 and num_images <= len(image_files):
        selected_files = random.sample(image_files, num_images)
    else:
        selected_files = image_files
    
    images = []
    for file in selected_files:
        try:
            img = Image.open(file).convert(type)
            img = img.resize(target_size) 
            images.append(img)
        except Exception as e:
            print(f'加载图片{file}出错:{e}')
    return images

if __name__ == "__main__":
    images = load_random_images("E:/code/watermark/code/ILSVRC/Data/DET/test", 10)
    print(images.shape)
    

            
            