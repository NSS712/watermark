import numpy as np
from skimage.metrics import structural_similarity as ssim
from attack import numpy_to_images

def z_check(images, k=5):
    n = images[0].shape[1] * images[0].shape[2] * images[0].shape[0]
    return np.array([abs(np.sum((img//k)%2==0)* 2 - n)/n for img in images])

def ssim_score(images_ori, images_watermark):
    ans = []
    images_ori = numpy_to_images(images_ori)
    images_watermark = numpy_to_images(images_watermark)
    for i in range(len(images_ori)):
        score, _ = ssim(np.array(images_ori[i].convert('L')), np.array(images_watermark[i].convert('L')), full=True)
        ans.append(score)
    return np.array(ans).mean()
    