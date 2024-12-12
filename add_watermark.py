import numpy as np

def modify_red_to_green(images, k=5):
    """
    将输入的图片,按照k个像素取值一组，将奇数组的像素全都映射到最近的偶数组。
    
    :param iamge: numpy类型的数组，可以是单张或者多组图片
    :param k: 每组的像素取值间隔
    :return 修改后的图像，numpy类型
    """
    pix_map = np.array(pixel_map(k=5))
    return pix_map[images]

def pixel_map(k=5):
    '''
    映射像素值，以k为分组，对x执行分区映射
    x 取值为0-255
    '''
    def map(x, k=5):
        if (x//k)%2 : # x属于红区
            if x%k <= k//2 or (x//k + 1)*k > 255: # 向下取整
                return max(0, x - x%k - 1)
            else:
                return (x//k + 1)*k
        else:
            return x
        
    return [map(x,k=k) for x in list(range(256))]

if __name__=="__main__":
    a = [list(range(20)) , list(range(235,255))]
    a = np.array(a)
    print(modify_red_to_green(a))
    # ans = [pixel_map(x,k=5) for x in a]    
    # for i in range(len(a)):
    #     print(f"{a[i]}->{ans[i]}")
