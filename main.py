from data import load_random_images
from add_watermark import modify_red_to_green
from attack import *
from examine import z_check
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from utils import draw
from examine import ssim_score
# 读n张图片进来
n = 1000
images_ori = load_random_images("E:/code/watermark/code/ILSVRC/Data/DET/test", 2*n)

# 红绿分区
images_watermark = modify_red_to_green(images_ori[:n],k=5)
img_ori_save = numpy_to_images(images_ori[:n])
img_water_save = numpy_to_images(images_watermark)
# for i, img in enumerate(img_ori_save[:n]):
#     img.save("./data/3/k/"+str(i)+".png")
# for i, img in enumerate(img_water_save[:n]):
#     img.save("./data/4/k/"+str(i)+".png")

# raise ValueError(1)
images = np.concatenate((images_watermark, images_ori[n:]), axis=0)
# 攻击
images_rotate = rotate_image(images, angle=90) # 旋转90度
images_crop = random_crop(images,area_ratio=0.5) # 剪裁0.5的面积

# 计算z检验
z_watermark = z_check(images)
z_rotate = z_check(images_rotate)
z_crop = z_check(images_crop)

# 计算 ssim


# 计算roc
y_true = np.array([1] * n + [0] * n)
fpr1, tpr1, _ = roc_curve(y_true, z_watermark)
auc1 = roc_auc_score(y_true, z_watermark)
fpr2, tpr2, _ = roc_curve(y_true, z_rotate)
auc2 = roc_auc_score(y_true, z_rotate)
fpr3, tpr3, _ = roc_curve(y_true, z_crop)
auc3 = roc_auc_score(y_true, z_crop)

draw((fpr1, tpr1, auc1), (fpr2, tpr2, auc2), (fpr3, tpr3, auc3), images[0], images_watermark[0], images_rotate[0], images_crop[0])

ssim = ssim_score(images_ori[:n], images_watermark)
print(ssim)




