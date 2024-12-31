import numpy as np
from attack import images_to_numpy

def map(x, k=5):
    if (x//k)%2 : # x属于红区
        if x%k < k//2 : # 向下取整
            return x - x%k - 1
        else:
            return (x//k + 1)*k
    else:
        return x

def modify_red_to_green(images, k=5, value_range=None):
    """
    将输入的图片,按照k个像素取值一组，将奇数组的像素全都映射到最近的偶数组。
    
    :param iamge: numpy类型的数组，可以是单张或者多组图片
    :param k: 每组的像素取值间隔
    :value_range: 需要映射为（0，255）的像素值，可以通过value_range参数指定 value_range=(0,255)
    """
    if value_range:
        pix_map = np.array(pixel_map(k=k, value_range=value_range))
    else:
        pix_map = np.vectorize(map)(images, k=k)
    return pix_map



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
    x = np.arange(-20, 20)
    print(x)
    y = modify_red_to_green(x, k=5)
    for i in range(len(x)):
        print(f"{x[i]} -> {y[i]}")
